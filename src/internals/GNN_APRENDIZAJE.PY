import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Importar la clase de aprendizaje que creamos anteriormente
class GNNHDBSCANLearner:
    """Sistema de aprendizaje incremental para GNN-HDBSCAN"""
    
    def __init__(self, learning_dir="gnn_hdbscan_learning", max_history=100):
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(exist_ok=True)
        self.max_history = max_history
        
        # Archivos de aprendizaje
        self.config_history_file = self.learning_dir / "config_history.json"
        self.best_models_dir = self.learning_dir / "best_models"
        self.best_models_dir.mkdir(exist_ok=True)
        self.embeddings_cache_dir = self.learning_dir / "embeddings_cache"
        self.embeddings_cache_dir.mkdir(exist_ok=True)
        
        # Cargar historial existente
        self.config_history = self._load_config_history()
        
    def _load_config_history(self):
        """Cargar historial de configuraciones probadas"""
        if self.config_history_file.exists():
            try:
                with open(self.config_history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_config_history(self):
        """Guardar historial de configuraciones"""
        if len(self.config_history) > self.max_history:
            self.config_history = sorted(
                self.config_history, 
                key=lambda x: x.get('score', -1), 
                reverse=True
            )[:self.max_history]
        
        with open(self.config_history_file, 'w') as f:
            json.dump(self.config_history, f, indent=2)
    
    def _get_data_fingerprint(self, data):
        """Crear huella digital de los datos para cache"""
        data_str = str(data.shape) + str(np.mean(data)) + str(np.std(data))
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def _get_promising_configs(self, top_k=5):
        """Obtener las configuraciones más prometedoras del historial"""
        if not self.config_history:
            return []
        
        sorted_configs = sorted(
            self.config_history, 
            key=lambda x: x.get('score', -1), 
            reverse=True
        )
        
        return sorted_configs[:top_k]
    
    def _adaptive_parameter_suggestion(self):
        """Sugerir parámetros adaptativos basados en el historial"""
        if len(self.config_history) < 3:
            return {
                'k_neighbors_range': [6, 8, 10, 12],
                'embedding_dims': [8, 12, 16, 20, 24],
                'hidden_dims': [16, 24, 32],
                'min_cluster_sizes': range(5, 31, 5),
                'min_samples_options': [3, 5, 7],
                'learning_rates': [0.001, 0.01, 0.02]
            }
        
        successful_configs = [c for c in self.config_history if c.get('score', 0) > 0.3]
        
        if successful_configs:
            k_neighbors = [c['params'].get('k_neighbors', 8) for c in successful_configs]
            embedding_dims = [c['params'].get('embedding_dim', 16) for c in successful_configs]
            
            k_range = range(max(4, min(k_neighbors)-2), min(16, max(k_neighbors)+3))
            dim_range = range(max(8, min(embedding_dims)-4), min(32, max(embedding_dims)+5), 4)
            
            return {
                'k_neighbors_range': list(k_range),
                'embedding_dims': list(dim_range),
                'hidden_dims': [16, 24, 32, 40],
                'min_cluster_sizes': range(5, 31, 5),
                'min_samples_options': [3, 5, 7, 9],
                'learning_rates': [0.005, 0.01, 0.015]
            }
        
        return {
            'k_neighbors_range': [4, 6, 8, 10, 12, 14],
            'embedding_dims': [6, 8, 12, 16, 20, 24, 28],
            'hidden_dims': [12, 16, 24, 32],
            'min_cluster_sizes': range(3, 35, 4),
            'min_samples_options': [2, 3, 5, 7],
            'learning_rates': [0.005, 0.01, 0.02, 0.03]
        }

    def apply_gnn_guided_hdbscan_with_learning(self, X_scaled, epochs=100, 
                                             min_size_range=(5, 31, 5), 
                                             use_cache=True, 
                                             exploration_ratio=0.3):
        """Aplicar clustering híbrido con aprendizaje incremental"""
        print("🧠 Aplicando GNN-HDBSCAN con Aprendizaje Incremental...")
        
        try:
            import hdbscan
            from sklearn.neighbors import kneighbors_graph
            from sklearn.metrics import silhouette_score
            
            data_fingerprint = self._get_data_fingerprint(X_scaled)
            print(f"   🔍 Huella digital de datos: {data_fingerprint}")
            
            promising_configs = self._get_promising_configs(top_k=3)
            adaptive_params = self._adaptive_parameter_suggestion()
            
            print(f"   📚 Configuraciones prometedoras encontradas: {len(promising_configs)}")
            
            best_result = None
            best_score = -1
            configs_tested = 0
            max_configs_to_test = 12
            
            configs_to_test = []
            
            # Configuraciones del historial (explotación)
            exploitation_count = int(max_configs_to_test * (1 - exploration_ratio))
            for config in promising_configs[:exploitation_count]:
                configs_to_test.append({
                    'source': 'history',
                    'k_neighbors': config['params'].get('k_neighbors', 8),
                    'embedding_dim': config['params'].get('embedding_dim', 16),
                    'hidden_dim': config['params'].get('hidden_dim', 24),
                    'lr': config['params'].get('lr', 0.01),
                    'expected_score': config.get('score', 0)
                })
            
            # Nuevas configuraciones (exploración)
            exploration_count = max_configs_to_test - len(configs_to_test)
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            
            for _ in range(exploration_count):
                configs_to_test.append({
                    'source': 'exploration',
                    'k_neighbors': np.random.choice(adaptive_params['k_neighbors_range']),
                    'embedding_dim': np.random.choice(adaptive_params['embedding_dims']),
                    'hidden_dim': np.random.choice(adaptive_params['hidden_dims']),
                    'lr': np.random.choice(adaptive_params['learning_rates']),
                    'expected_score': 0
                })
            
            print(f"   🔬 Probando {len(configs_to_test)} configuraciones...")
            
            # Evaluación de configuraciones
            for i, config in enumerate(configs_to_test):
                try:
                    print(f"   ⚙️  Config {i+1}/{len(configs_to_test)} "
                          f"({config['source']}): k={config['k_neighbors']}, "
                          f"dim={config['embedding_dim']}, lr={config['lr']}")
                    
                    # Cache de embeddings
                    embedding_cache_key = f"{data_fingerprint}_{config['k_neighbors']}_{config['embedding_dim']}_{config['hidden_dim']}"
                    embedding_cache_file = self.embeddings_cache_dir / f"{embedding_cache_key}.pkl"
                    
                    embeddings = None
                    if use_cache and embedding_cache_file.exists():
                        print("      📦 Cargando embeddings desde cache...")
                        try:
                            with open(embedding_cache_file, 'rb') as f:
                                cache_data = pickle.load(f)
                                embeddings = cache_data['embeddings']
                                print(f"      ✅ Cache hit")
                        except:
                            print("      ⚠️  Error cargando cache, regenerando...")
                    
                    if embeddings is None:
                        embeddings = self._generate_gnn_embeddings(
                            X_scaled, 
                            config['k_neighbors'],
                            config['embedding_dim'],
                            config['hidden_dim'],
                            config['lr'],
                            epochs
                        )
                        
                        if embeddings is not None and use_cache:
                            try:
                                cache_data = {
                                    'embeddings': embeddings,
                                    'config': config,
                                    'timestamp': datetime.now().isoformat()
                                }
                                with open(embedding_cache_file, 'wb') as f:
                                    pickle.dump(cache_data, f)
                                print("      💾 Embeddings guardados en cache")
                            except Exception as e:
                                print(f"      ⚠️  Error guardando cache: {e}")
                    
                    if embeddings is None:
                        continue
                    
                    clustering_result = self._apply_intelligent_hdbscan(
                        embeddings, X_scaled, config, min_size_range
                    )
                    
                    if clustering_result and clustering_result['score'] > best_score:
                        best_score = clustering_result['score']
                        best_result = clustering_result
                        print(f"      🎉 ¡Nueva mejor configuración! Score: {best_score:.4f}")
                    
                    self._save_config_result(config, clustering_result, data_fingerprint)
                    configs_tested += 1
                    
                except Exception as e:
                    print(f"      ❌ Error en configuración {i+1}: {str(e)}")
                    continue
            
            if best_result:
                print(f"\n✅ GNN-HDBSCAN con Aprendizaje Completado")
                print(f"   🏆 Mejor Score: {best_result['score']:.4f}")
                print(f"   📊 Clusters: {best_result['n_clusters']}")
                print(f"   🔧 Configuraciones probadas: {configs_tested}")
                print(f"   📚 Total en historial: {len(self.config_history)}")
                
                if best_result['score'] > 0.6:
                    self._save_best_model(best_result, data_fingerprint)
                
                return best_result
            else:
                print("❌ No se encontraron configuraciones válidas")
                return None
                
        except ImportError as e:
            print("❌ Error: Librerías requeridas no disponibles")
            print("   💡 Instalación requerida: pip install torch hdbscan scikit-learn")
            return None
        except Exception as e:
            print(f"❌ Error general: {str(e)}")
            return None
    
    def _generate_gnn_embeddings(self, X_scaled, k_neighbors, embedding_dim, 
                                hidden_dim, lr, epochs):
        """Generar embeddings usando GNN"""
        try:
            from sklearn.neighbors import kneighbors_graph
            
            knn_graph = kneighbors_graph(
                X_scaled, 
                n_neighbors=k_neighbors, 
                mode='connectivity',
                include_self=False
            )
            
            edges = np.array(knn_graph.nonzero())
            edge_index = torch.tensor(edges, dtype=torch.long)
            node_features = torch.tensor(X_scaled, dtype=torch.float32)
            
            class OptimizedEmbeddingGCN(nn.Module):
                def __init__(self, input_dim, hidden_dim, embedding_dim):
                    super(OptimizedEmbeddingGCN, self).__init__()
                    
                    self.conv1 = nn.Linear(input_dim, hidden_dim)
                    self.conv2 = nn.Linear(hidden_dim, embedding_dim)
                    self.conv3 = nn.Linear(embedding_dim, embedding_dim)
                    
                    self.dropout = nn.Dropout(0.1)
                    self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                    self.batch_norm2 = nn.BatchNorm1d(embedding_dim)
                    
                    nn.init.xavier_uniform_(self.conv1.weight)
                    nn.init.xavier_uniform_(self.conv2.weight)
                    nn.init.xavier_uniform_(self.conv3.weight)
                
                def forward(self, x, edge_index):
                    x1 = self.message_passing(x, edge_index, self.conv1)
                    x1 = self.batch_norm1(x1)
                    x1 = F.leaky_relu(x1, 0.2)
                    x1 = self.dropout(x1)
                    
                    x2 = self.message_passing(x1, edge_index, self.conv2)
                    x2 = self.batch_norm2(x2)
                    x2 = F.leaky_relu(x2, 0.2)
                    
                    embeddings = self.message_passing(x2, edge_index, self.conv3)
                    return F.normalize(embeddings, p=2, dim=1)
                
                def message_passing(self, x, edge_index, linear_layer):
                    row, col = edge_index
                    x_transformed = linear_layer(x)
                    out = torch.zeros_like(x_transformed)
                    
                    for i in range(x.size(0)):
                        neighbors = col[row == i]
                        if len(neighbors) > 0:
                            self_feature = x_transformed[i:i+1]
                            neighbor_features = x_transformed[neighbors]
                            
                            similarities = F.cosine_similarity(
                                self_feature.repeat(len(neighbors), 1), 
                                neighbor_features,
                                dim=1
                            )
                            weights = F.softmax(similarities, dim=0)
                            
                            weighted_neighbors = torch.sum(
                                neighbor_features * weights.unsqueeze(1), dim=0
                            )
                            out[i] = 0.7 * x_transformed[i] + 0.3 * weighted_neighbors
                        else:
                            out[i] = x_transformed[i]
                    
                    return out
            
            model = OptimizedEmbeddingGCN(
                input_dim=node_features.shape[1],
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim
            )
            
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            
            model.train()
            best_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                embeddings = model(node_features, edge_index)
                
                local_loss = self._compute_local_structure_loss(embeddings, edge_index)
                separation_loss = self._compute_cluster_separation_loss(embeddings, k=6)
                
                total_loss = local_loss + 0.2 * separation_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            model.eval()
            with torch.no_grad():
                final_embeddings = model(node_features, edge_index)
                return final_embeddings.numpy()
                
        except Exception as e:
            print(f"      ❌ Error generando embeddings: {str(e)}")
            return None
    
    def _apply_intelligent_hdbscan(self, embeddings, _, config, min_size_range):
        """Aplicar HDBSCAN con parámetros inteligentes"""
        try:
            import hdbscan
            from sklearn.metrics import silhouette_score
            
            best_score = -1
            best_labels = None
            best_params = None
            
            min_sizes = list(range(min_size_range[0], min_size_range[1], min_size_range[2]))
            min_samples_options = [3, 5, 7]
            
            if len(min_sizes) * len(min_samples_options) > 12:
                min_sizes = min_sizes[::2]
            
            for min_size in min_sizes:
                for min_samples in min_samples_options:
                    try:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_size,
                            min_samples=min_samples,
                            cluster_selection_method='leaf',
                            metric='euclidean'
                        )
                        
                        labels = clusterer.fit_predict(embeddings)
                        
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        noise_ratio = n_noise / len(labels)
                        
                        if n_clusters >= 3 and noise_ratio < 0.5 and n_clusters <= 15:
                            try:
                                score = silhouette_score(embeddings, labels)
                                if score > best_score:
                                    best_score = score
                                    best_labels = labels
                                    best_params = {
                                        'min_cluster_size': min_size,
                                        'min_samples': min_samples,
                                        'n_clusters': n_clusters,
                                        'n_noise': n_noise,
                                        'noise_ratio': noise_ratio
                                    }
                            except:
                                continue
                    except:
                        continue
            
            if best_labels is not None:
                return {
                    'score': best_score,
                    'labels': best_labels,
                    'n_clusters': best_params['n_clusters'],
                    'params': {**config, **best_params},
                    'embeddings': embeddings
                }
            
            return None
            
        except Exception as e:
            print(f"      ❌ Error en HDBSCAN: {str(e)}")
            return None
    
    def _save_config_result(self, config, result, data_fingerprint):
        """Guardar resultado de configuración en historial"""
        if result is None:
            return
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'data_fingerprint': data_fingerprint,
            'params': config,
            'score': result['score'],
            'n_clusters': result['n_clusters'],
            'clustering_params': result.get('params', {}),
            'source': config.get('source', 'unknown')
        }
        
        self.config_history.append(history_entry)
        self._save_config_history()
    
    def _save_best_model(self, result, data_fingerprint):
        """Guardar modelo excepcionalmente bueno"""
        try:
            model_file = self.best_models_dir / f"best_model_{data_fingerprint}_{result['score']:.4f}.pkl"
            
            model_data = {
                'timestamp': datetime.now().isoformat(),
                'score': result['score'],
                'params': result['params'],
                'n_clusters': result['n_clusters'],
                'embeddings': result.get('embeddings'),
                'labels': result['labels']
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"      💎 Modelo excepcional guardado: {model_file.name}")
            
        except Exception as e:
            print(f"      ⚠️  Error guardando modelo: {e}")
    
    def _compute_local_structure_loss(self, embeddings, edge_index):
        """Loss para preservar estructura local"""
        row, col = edge_index
        edge_distances = torch.norm(embeddings[row] - embeddings[col], dim=1)
        return torch.mean(edge_distances)
    
    def _compute_cluster_separation_loss(self, embeddings, k=6):
        """Loss para fomentar separación entre clusters"""
        try:
            from sklearn.cluster import KMeans
            
            with torch.no_grad():
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
                temp_labels = kmeans.fit_predict(embeddings.detach().numpy())
                
                centers = torch.zeros(k, embeddings.shape[1])
                for i in range(k):
                    mask = temp_labels == i
                    if mask.sum() > 0:
                        centers[i] = embeddings[mask].mean(dim=0)
            
            center_distances = torch.cdist(centers, centers)
            mask = torch.eye(k, dtype=torch.bool)
            center_distances = center_distances.masked_fill(mask, float('inf'))
            min_distances = torch.min(center_distances, dim=1)[0]
            
            return -torch.mean(min_distances)
            
        except:
            return torch.tensor(0.0)
    
    def get_learning_stats(self):
        """Obtener estadísticas del aprendizaje"""
        if not self.config_history:
            return "No hay historial de aprendizaje disponible."
        
        scores = [entry.get('score', 0) for entry in self.config_history]
        best_entry = max(self.config_history, key=lambda x: x.get('score', 0))
        
        stats = f"""
📊 Estadísticas de Aprendizaje GNN-HDBSCAN:
   🎯 Configuraciones probadas: {len(self.config_history)}
   🏆 Mejor score: {max(scores):.4f}
   📈 Score promedio: {np.mean(scores):.4f}
   📉 Score mínimo: {min(scores):.4f}
   🕒 Última actualización: {self.config_history[-1]['timestamp'][:19]}
   
🏅 Mejor configuración:
   📅 Fecha: {best_entry['timestamp'][:19]}
   🎯 Score: {best_entry['score']:.4f}
   📊 Clusters: {best_entry['n_clusters']}
   ⚙️  Parámetros: {best_entry['params']}
        """
        
        return stats
    
    def clear_learning_history(self):
        """Limpiar historial de aprendizaje"""
        self.config_history = []
        self._save_config_history()
        
        for cache_file in self.embeddings_cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        for model_file in self.best_models_dir.glob("*.pkl"):
            model_file.unlink()
        
        print("🧹 Historial de aprendizaje limpiado")


