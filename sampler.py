import torch

@torch.no_grad()
def multistep_consistency_sampling(net, latents, t_steps):   
    t_steps = torch.as_tensor(t_steps).to(latents.device)
    latents = latents * t_steps[0]
    x = net(latents, t_steps[0])
    
    for t in t_steps[1:]:
        z = torch.randn_like(x)
        x_tn = x + (t ** 2 - net.sigma_min ** 2).sqrt() * z
        x = net(x_tn, t)
    
    return x
