import torch
import torch.nn as nn
import numpy as np

class CEMLinearPolicy(nn.Module):
    def __init__(self, sequence_length=10, num_sensors=9, num_actions=3):
        super(CEMLinearPolicy, self).__init__()
        
        # Dimension de l'état aplati : 10 * 9 = 90
        self.input_dim = sequence_length * num_sensors
        self.num_actions = num_actions
        
        # La matrice W et le biais b (notre "régression logistique")
        self.fc = nn.Linear(self.input_dim, self.num_actions)
        
        # DÉSACTIVATION RIGOUREUSE DU GRADIENT
        # Indispensable pour CEM : on économise de la RAM et du temps de calcul
        # car PyTorch n'a plus besoin de construire le graphe de calcul dynamique.
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Calcule l'action déterministe à partir de l'observation.
        """
        # Conversion numpy -> tensor si nécessaire
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # 1. Aplatissement strict de la matrice [10, 9] en un vecteur [90]
        if x.dim() == 2:  # Cas d'un seul environnement (10, 9)
            x = x.view(-1)
        elif x.dim() == 3: # Cas vectorisé (Batch, 10, 9)
            x = x.view(x.size(0), -1)

        # 2. Produit matriciel : logits = W * x + b
        logits = self.fc(x)
        
        # 3. Argmax pour choisir l'action (déterministe, pas besoin de Softmax ici)
        if logits.dim() == 1:
            return torch.argmax(logits).item()
        else:
            return torch.argmax(logits, dim=1).numpy()

    
    def get_weights(self) -> np.ndarray:
        """
        Extrait la matrice W et le biais b pour les concaténer en un seul vecteur thêta 1D.
        Dimension attendue : (90 * 3) + 3 = 273 paramètres.
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, theta: np.ndarray):
        """
        Prend un vecteur thêta 1D (muté par l'algorithme CEM) et l'injecte
        rigoureusement dans la matrice W et le biais b du réseau.
        """
        theta_tensor = torch.tensor(theta, dtype=torch.float32)
        pointer = 0
        
        for param in self.parameters():
            # Nombre d'éléments dans ce paramètre (ex: 270 pour W, 3 pour b)
            num_param = param.numel() 
            
            # Extraction du sous-vecteur correspondant
            param_update = theta_tensor[pointer : pointer + num_param]
            
            # Reshape et copie en place dans les poids du réseau
            param.data.copy_(param_update.view_as(param.data))
            
            pointer += num_param