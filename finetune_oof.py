"""Out-of-fold fine-tuning of a pretrained emotion classifier on the competition's 6 labels.
Run with the separate GPU environment. Reads and writes only data/kaggle/ft/."""
import sys, json, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

short, model_name = sys.argv[1], sys.argv[2]
FT_DIR = "data/kaggle/ft"
cfg = json.load(open(f"{FT_DIR}/inputs.json"))
LABELS, SEED = cfg["labels"], cfg["seed"]
train_text, test_text = cfg["train_text"], cfg["test_text"]
y = np.array([LABELS.index(l) for l in cfg["train_label"]])
fold = np.array(cfg["fold"])
P, TAG = cfg["params"], cfg["tag"]
EPOCHS, LR, BATCH, MAX_LEN = P["epochs"], P["lr"], P["batch"], P["max_len"]
print(f"{short}: settings {P}", flush=True)
device = "cuda"
tok = AutoTokenizer.from_pretrained(model_name)

def batches(texts, labels=None, shuffle=False, seed=0):
    """Tokenized mini-batches on the GPU; shuffled with a fixed seed when training."""
    idx = np.arange(len(texts))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        enc = tok([texts[j] for j in b], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor(labels[b])
        yield enc.to(device)

@torch.no_grad()
def predict(model, texts):
    model.eval()
    out = []
    for enc in batches(texts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.vstack(out)

def macro_f1(y_true, y_pred, n=len(LABELS)):
    """Mean of per-class F1 (numpy only, no sklearn in this environment)."""
    f1s = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        prec = tp / max(np.sum(y_pred == c), 1)
        rec = tp / max(np.sum(y_true == c), 1)
        f1s.append(0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))

oof = np.zeros((len(train_text), len(LABELS)), dtype=np.float32)
test_sum = np.zeros((len(test_text), len(LABELS)), dtype=np.float32)
n_folds = len(set(fold))
for k in range(n_folds):
    t0 = time.time()
    torch.manual_seed(SEED + k)
    tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
    # New 6-class head on the pretrained body. problem_type overrides SamLowe's
    # multi-label setting, so the loss is softmax cross-entropy for one label per post.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
        problem_type="single_label_classification", ignore_mismatched_sizes=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    tr_text = [train_text[i] for i in tr]
    for epoch in range(EPOCHS):
        model.train()
        total, n_batches = 0.0, 0
        for enc in batches(tr_text, y[tr], shuffle=True, seed=SEED * 100 + k * 10 + epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            n_batches += 1
        print(f"{short} fold {k} epoch {epoch}: mean train loss {total / n_batches:.3f}, train loss {total:.1f}", flush=True)
        ep_pred = predict(model, [train_text[i] for i in va]).argmax(1)
        print(f"{short} fold {k} epoch {epoch}: OOF macro F1 {macro_f1(y[va], ep_pred):.3f}", flush=True)
        model.train()  # predict() switched to eval mode
    oof[va] = predict(model, [train_text[i] for i in va])   # only this fold: unseen by this model
    va_pred = oof[va].argmax(1)
    print(f"{short} fold {k}: OOF accuracy {(va_pred == y[va]).mean():.3f}, "
          f"macro F1 {macro_f1(y[va], va_pred):.3f}, {time.time() - t0:.0f}s, "
          f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    test_sum += predict(model, test_text)
    del model, opt
    torch.cuda.empty_cache()

np.save(f"{FT_DIR}/{short}_{TAG}_oof_train.npy", oof)
np.save(f"{FT_DIR}/{short}_{TAG}_test.npy", test_sum / n_folds)
print(f"{short}: OOF macro F1 over all folds {macro_f1(y, oof.argmax(1)):.3f}")
print("saved", short)
