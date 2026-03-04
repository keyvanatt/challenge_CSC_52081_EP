import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEnginePolicy(nn.Module):
    def __init__(self, num_sensors=9, num_actions=3):
        super(CNNEnginePolicy, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_sensors, out_channels=32, kernel_size=3)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
    
        self.flattened_size = 64 * 6
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        # 1. Gestion stricte des dimensions
        # Entrée x de l'environnement : ex. [40, 9] -> (BatchxTemps, Capteurs)
        # ou [10, 9] -> (Temps, Capteurs)
    
        x = x.view(-1, 10, 9)
        # Maintenant x est de la forme (Batch, Temps, Capteurs) -> [4, 10, 9] si batch de 4 et 10 timesteps, sinon [1, 10, 9] si un seul batch,  ou [1, 1, 9] si un seul batch et un seul timestep.
            
        # On passe de (Batch, Temps, Capteurs) à (Batch, Capteurs, Temps) -> [4, 9, 10]

        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 4. Aplatissement (Flatten) pour passer aux couches denses
        x = x.view(x.size(0), -1) # Sortie : [4, 384]
        
        # 5. Réseau de décision final
        x = F.relu(self.fc1(x))
        action_values = self.fc2(x) # Sortie : [4, 3]
        
        return action_values

def init_weights_biased(m):
    if isinstance(m, nn.Linear):
        # Initialisation standard des poids
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
        # On met tous les biais à 0
        nn.init.constant_(m.bias, 0)
        
        # Si c'est la couche de sortie, on booste l'action 0 (Do Nothing)
        if m.out_features == 3:
            with torch.no_grad():
                # On donne un avantage numérique à l'action 0
                # "Ne rien faire" devient l'action par défaut
                m.bias[0] = 1.0  
                # On peut même pénaliser légèrement la vente pour éviter l'arrêt précoce
                m.bias[2] = -1.0 


if __name__ == "__main__":
    mock_obs = torch.randn(40, 9) 
    model = CNNEnginePolicy()
    preds = model(mock_obs)
    
    print(f"Entrée : {mock_obs.shape} -> Sortie : {preds.shape}") 
    # Doit donner Entrée : torch.Size([40, 9]) -> Sortie : torch.Size([4, 3])