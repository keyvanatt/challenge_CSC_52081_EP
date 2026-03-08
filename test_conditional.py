import numpy as np
import torch
from models.conditionnal_model import ConditionalPolicy
from student_client import create_student_gym_env

def test_agent(checkpoint_path="saves/cem_checkpoint.npz", num_episodes=20):
    # 1. Initialisation stricte (Un seul environnement pour l'évaluation)
    env = create_student_gym_env()
    agent = ConditionalPolicy(num_sensors=9, num_actions=3)
    
    # ==========================================
    # 2. CHARGEMENT RIGOUREUX DU CHECKPOINT
    # ==========================================
    try:
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        best_weights = checkpoint['best_theta']
        
        # Vérification mathématique du cas "None" encapsulé par Numpy
        if best_weights.shape == () or best_weights is None:
            print("Aucun record global (best_theta) trouvé dans l'archive.")
            print("Utilisation du vecteur moyen actuel (mu) par défaut.")
            best_weights = checkpoint['mu']
        else:
            print(f"Record (best_theta) chargé avec succès depuis la génération {checkpoint['iteration'] + 1}.")
            print(f"Score associé dans l'entraînement : {checkpoint['best_reward']:.1f}")
            
        agent.set_weights(best_weights)
        print("-" * 65)
        
    except FileNotFoundError:
        print(f"ERREUR FATALE : Le fichier '{checkpoint_path}' est introuvable.")
        return
    except Exception as e:
        print(f"ERREUR FATALE lors de la lecture de l'archive : {e}")
        return

    # 3. Variables de suivi mathématique
    total_rewards = []
    failures = 0
    total_repairs = []
    total_sells = []
    sell_steps = []

    print("\n=== DÉBUT DE L'ÉVALUATION OFFICIELLE ===")
    print(f"{'Épisode':<8} | {'Statut':<13} | {'Récompense':<10} | {'Réparations':<11} | {'Fin au Step'}")
    print("-" * 65)

    for ep in range(num_episodes):
        state, _ = env.reset()
        agent.reset()
        
        # Sécurisation de la dimension mathématique (10, 9)
        if state.ndim == 1 and state.shape[0] == 9:
            state = np.tile(state, (10, 1))     
        
        done = False
        step = 0
        ep_reward = 0.0
        ep_repairs = 0
        sold = False
        sell_step = -1
        
        while not done:
            action = agent(state) 
            print(f"Action choisie : {action} (step {step})")
            if action == 1:
                ep_repairs += 1
            elif action == 2:
                sold = True
                sell_step = step
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            step += 1
            
            done = terminated or truncated

            print(f"Reward reçu : {reward:.1f} | Total reward : {ep_reward:.1f} | Done : {done}")
            
            if not done:
                state = next_state
        
        # 4. Diagnostic rigoureux de la fin de vol
        failure = False
        if not sold and terminated: 
            failure = True
            
        if failure:
            failures += 1

        # 5. Stockage des métriques
        total_rewards.append(ep_reward)
        total_repairs.append(ep_repairs)
        
        if sold:
            total_sells.append(1)
            sell_steps.append(sell_step)
        else:
            total_sells.append(0)
            
        # 6. Affichage en direct
        status = "VENDU" if sold else ("CRASH" if failure else "LIMITE TEMPS")
        print(f"Vol {ep+1:02d}   | {status:13s} | {ep_reward:<10.1f} | {ep_repairs:<11} | {step}")

    # ==========================================
    # CALCUL DES MOYENNES MATHÉMATIQUES
    # ==========================================
    mean_reward = np.mean(total_rewards)
    failure_rate = (failures / num_episodes) * 100
    mean_repairs = np.mean(total_repairs)
    mean_sells = np.mean(total_sells)
    mean_sell_step = np.mean(sell_steps) if len(sell_steps) > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"RÉSULTATS MATHÉMATIQUES GLOBAUX (Sur {num_episodes} épisodes)")
    print("=" * 50)
    print(f"Récompense moyenne      : {mean_reward:.2f}")
    print(f"Taux de Failure         : {failure_rate:.1f}%")
    print(f"Réparations moyennes    : {mean_repairs:.2f} par vol")
    print(f"Ventes moyennes         : {mean_sells:.2f} par vol")
    print(f"Étape moyenne de vente  : {mean_sell_step:.1f}")

    env.close()

if __name__ == "__main__":
    test_agent(

        num_episodes=100
    )