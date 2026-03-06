import torch
import torch.nn as nn
import numpy as np
import json

class ConditionalPolicy(nn.Module):
    def __init__(self, num_sensors=9, num_actions=3):
        super(ConditionalPolicy, self).__init__()
        
        self.input_dim = num_sensors
        self.num_actions = num_actions
        
        
        self.normal_values = nn.Parameter(torch.ones(num_sensors, dtype=torch.float32), requires_grad=False)
        self.sensor_weights = nn.Parameter(torch.ones(num_sensors, dtype=torch.float32), requires_grad=False)
        
        self.no_repair_confidence = nn.Parameter(torch.tensor(0.25, dtype=torch.float32), requires_grad=False)
        self.no_sell_confidence = nn.Parameter(torch.tensor(0.25, dtype=torch.float32), requires_grad=False)
        
        self.repair_every = nn.Parameter(torch.tensor(5.0, dtype=torch.float32), requires_grad=False)
        self.sell_after = nn.Parameter(torch.tensor(25.0, dtype=torch.float32), requires_grad=False)
        
        self.current_step = 0
        self.last_repair_step = 0

    def _in_confidence_zone(self, x_mean, confidence_threshold):
        deviation = torch.abs(x_mean - self.normal_values)
        
        weighted_deviation = deviation * self.sensor_weights
        
        return weighted_deviation.sum().item() <= confidence_threshold

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        assert x.dim() == 2 and x.shape[0] == 10 and x.shape[1] == 9, f"Expected input of shape (10, 9), got {x.shape}"
        
        x_mean = x.mean(dim=0)  # (sensors,)
        action = 0


        sell_time = max(1, int(self.sell_after.item()))
        repair_time = max(1, int(self.repair_every.item()))

        if self.current_step >= sell_time:
            if not self._in_confidence_zone(x_mean, self.no_sell_confidence.item()):
                action = 2  # Vendre
        elif (self.current_step - self.last_repair_step) >= repair_time:
            if not self._in_confidence_zone(x_mean, self.no_repair_confidence.item()):
                action = 1  # Réparer
                self.last_repair_step = self.current_step
                
        self.current_step += 1
        return action
    
    def reset(self):
        """Réinitialise les horloges pour un nouveau vol"""
        self.current_step = 0
        self.last_repair_step = 0
    
    def get_weights(self) -> np.ndarray:
        weights = [
            self.normal_values.data.cpu().numpy(),          # Index 0 à 8
            self.sensor_weights.data.cpu().numpy(),         # Index 9 à 17
            np.array([self.no_repair_confidence.item()]),   # Index 18
            np.array([self.no_sell_confidence.item()]),     # Index 19
            np.array([self.repair_every.item()]),           # Index 20
            np.array([self.sell_after.item()])              # Index 21
        ]
        return np.concatenate(weights)

    def set_weights(self, theta: np.ndarray):
        # Extraction stricte selon la taille mathématique des tenseurs
        self.normal_values.data.copy_(torch.tensor(theta[0:9], dtype=torch.float32))
        self.sensor_weights.data.copy_(torch.tensor(theta[9:18], dtype=torch.float32))
        self.no_repair_confidence.data.copy_(torch.tensor(theta[18], dtype=torch.float32))
        self.no_sell_confidence.data.copy_(torch.tensor(theta[19], dtype=torch.float32))
        self.repair_every.data.copy_(torch.tensor(theta[20], dtype=torch.float32))
        self.sell_after.data.copy_(torch.tensor(theta[21], dtype=torch.float32))

    def initialize_weights(self, jsonl_datapath: str, first_n_steps: int = 5):
        import json
        episodes = []
        with open(jsonl_datapath, 'r') as f:
            episodes.extend([json.loads(line) for line in f])

        # Listes pour stocker toutes les lectures "normales"
        normal_readings = []
        
        for episode in episodes:
            obs_seq = episode["obs"]
            steps_to_consider = min(len(obs_seq), first_n_steps)
            
            for i in range(steps_to_consider):
                # Extraction stricte de la lecture du capteur
                reading = obs_seq[i]
                normal_readings.append(reading)
                
        if len(normal_readings) > 0:
            # Conversion en matrice (N_samples, num_sensors)
            readings_matrix = torch.tensor(normal_readings, dtype=torch.float32)
            
            # 1. Calcul des moyennes (normal_values)
            mean_parameters = readings_matrix.mean(dim=0)
            self.normal_values.data.copy_(mean_parameters)
            
            # 2. Calcul des écart-types (standard deviation)
            std_parameters = readings_matrix.std(dim=0)
            
            # 3. Pondération par l'inverse de l'écart-type (avec epsilon de sécurité)
            epsilon = 1e-6
            intelligent_weights = 1.0 / (std_parameters + epsilon)
            
            # 4. Normalisation mathématique des poids
            # Pour que la somme des poids fasse toujours num_sensors (ex: 9)
            # Cela évite de dérégler complètement tes seuils de "confidence" actuels
            intelligent_weights = intelligent_weights / intelligent_weights.mean()
            
            self.sensor_weights.data.copy_(intelligent_weights)
            
            print(f"Initialisation réussie sur {len(normal_readings)} pas de temps.")
            print(f"Moyennes calculées : {mean_parameters.numpy()}")
            print(f"Poids intelligents (Inverse Variance) : {intelligent_weights.numpy()}")

    def __str__(self) -> str:
        return (super().__str__() + 
                f"\nNormal Values: {self.normal_values.data.cpu().numpy()}" +
                f"\nSensor Weights: {self.sensor_weights.data.cpu().numpy()}" +
                f"\nNo Repair Confidence: {self.no_repair_confidence.item():.4f}" +
                f"\nNo Sell Confidence: {self.no_sell_confidence.item():.4f}" +
                f"\nRepair Every: {self.repair_every.item():.1f} steps" +
                f"\nSell After: {self.sell_after.item():.1f} steps")

    def __repr__(self) -> str:
        return self.__str__()