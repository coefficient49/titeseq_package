import torch
import torch.nn as nn

class TiteSeqModel(nn.Module):
    def __init__(self, num_variants, b_bg):
        super().__init__()
        # Use log-space parameters for numerical stability
        self.log_kd = nn.Parameter(torch.full((num_variants, 1), -7.0))
        self.log_a = nn.Parameter(torch.full((num_variants, 1), 10.0))
        self.b_bg = b_bg

    def forward(self, c):
        kd = torch.exp(self.log_kd)
        a = torch.exp(self.log_a)
        # Batch Hill function calculation
        return a * (c / (c + kd)) + self.b_bg

def fit_gpu(counts_tensor, concs, b_bg, epochs=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TiteSeqModel(counts_tensor.shape[0], b_bg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # Using PoissonNLLLoss to represent Equation A3 [cite: 975]
    criterion = nn.PoissonNLLLoss(log_input=False, full=True)
    
    c = torch.tensor(concs, device=device).float()
    y = counts_tensor.to(device).float()

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(c)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
    return torch.exp(model.log_kd).detach().cpu().numpy(), \
           torch.exp(model.log_a).detach().cpu().numpy()