"""Small three-class temporal model over the baseline's observation features.

Only tracking-derived features enter the network. Unknown targets have zero
loss weight. Entire windows, including context, use one corruption variant.
"""
from __future__ import annotations

import numpy as np
from experiments import learned_temporal as lt, supervised


def network(n_features):
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence = nn.GRU(n_features, 24, batch_first=True, bidirectional=True)
            self.output = nn.Linear(48, 3)

        def forward(self, x):
            return self.output(self.sequence(x)[0])

    return Network()


def fit(records):
    import torch
    from torch.utils.data import DataLoader, Dataset
    torch.set_num_threads(1)
    torch.manual_seed(lt.SEED)
    mixed = [dict(r, x=np.where(r['replace_rows'][:, None], r['alternate'], r['x']))
             for r in records]
    median, scale = lt.normalization(mixed)
    del mixed
    original = [lt.normalize(r['x'], median, scale) for r in records]
    alternate = [lt.normalize(r['alternate'], median, scale) for r in records]
    samples = [(i, a, size, full, use) for i, r in enumerate(records)
               for (a, size, full), use in zip(lt.chunks(r), r['replace_windows'])]
    masses = np.array([1/(sum(v['source'] == r['source'] for v in records)
                          * max(1, np.sum(r['y'] >= 0))) for r in records])
    masses /= masses.mean()

    class Windows(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            i, a, size, full, use = samples[index]
            target = np.zeros(lt.CORE, np.int64)
            weight = np.zeros(lt.CORE, np.float32)
            y = records[i]['y'][a:a+size]
            target[:size] = np.maximum(y, 0)
            weight[:size] = (y >= 0)*masses[i]
            return (alternate if use else original)[i][full], target, weight

    loader = DataLoader(Windows(), batch_size=64, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(lt.SEED))
    model = network(len(median)*2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history, updates = [], 0
    for epoch in range(lt.EPOCHS):
        model.train()
        numerator = denominator = 0.
        for x, y, weight in loader:
            if not weight.sum():
                continue
            optimizer.zero_grad()
            logits = model(x)[:, lt.CONTEXT:lt.CONTEXT+lt.CORE]
            losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, reduction='none')
            loss = (losses*weight).sum()/weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            updates += 1
            numerator += float((losses.detach()*weight).sum())
            denominator += float(weight.sum())
        history.append(numerator/denominator)
        print('HAD GRU', epoch+1, round(history[-1], 5), flush=True)
    return dict(kind='had_bigru', state={k: v.detach().numpy() for k, v in model.state_dict().items()},
                median=median, scale=scale, history=history, input_features=len(median)*2,
                feature_names=records[0]['names'], windows_per_epoch=len(samples),
                optimizer_updates=updates, parameters=sum(p.numel() for p in model.parameters()))


def predict(record, model):
    """Symmetrize H/A probabilities, as the existing team HGB does."""
    import torch
    if record['names'] != model['feature_names']:
        raise ValueError('Feature schema differs')
    torch.set_num_threads(1)
    net = network(model['input_features'])
    net.load_state_dict({k: torch.as_tensor(v) for k, v in model['state'].items()})
    net.eval()
    windows = list(lt.chunks(record))
    results = []
    for swap in (False, True):
        raw = supervised.swap_team_features(record['x'], record['names']) if swap else record['x']
        x = lt.normalize(raw, model['median'], model['scale'])
        out = np.empty((len(x), 3), np.float32)
        with torch.inference_mode():
            for offset in range(0, len(windows), 64):
                batch = windows[offset:offset+64]
                probabilities = torch.softmax(net(torch.as_tensor(np.stack([x[f] for _, _, f in batch]))), -1).numpy()
                for i, (a, size, _) in enumerate(batch):
                    out[a:a+size] = probabilities[i, lt.CONTEXT:lt.CONTEXT+size]
        results.append(out[:, [1, 0, 2]] if swap else out)
    return (results[0]+results[1])*.5
