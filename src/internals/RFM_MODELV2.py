# Proyecto de Clusterización en la Segmentación de Clientes Dinámica
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
warnings.filterwarnings('ignore')
from scipy import stats
import torch


# Librerías de clustering
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz
import hdbscan
from sklearn.cluster import SpectralClustering, Birch, AffinityPropagation, AgglomerativeClustering, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
from skfuzzy import control as ctrl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Preprocesamiento y métricas
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Progreso
from tqdm import tqdm
from itertools import product
import re


class RFMClusteringAnalysis:
    """Clase para análisis completo de clustering en datos RFM"""
    
    def __init__(self, data_path=None, df=None):
        """
        Inicializar con datos desde archivo CSV o DataFrame
        """
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path, sep=None, engine="python")
        else:
            # Cargar datos por defecto - ajustado para la estructura del repo
            current_path = Path.cwd()
            
            # Buscar el archivo Data.csv en diferentes ubicaciones posibles
            possible_paths = [
                current_path / "Data" / "Data.csv",  # Si estás en la raíz
                current_path.parent / "Data" / "Data.csv",  # Si estás en src/
                current_path.parent.parent / "Data" / "Data.csv",  # Si estás en src/subfolder/
                current_path / ".." / "Data" / "Data.csv"  # Ruta relativa
            ]
            
            ruta_csv = None
            for path in possible_paths:
                if path.exists():
                    ruta_csv = path
                    break
            
            if ruta_csv is None:
                raise FileNotFoundError("No se encontró el archivo Data.csv. Verifica la estructura del repositorio.")
            
            self.df = pd.read_csv(ruta_csv, sep=None, engine="python")
            print(f"📁 Datos cargados desde: {ruta_csv}")
        
        self.df_rfm = None
        self.X_scaled = None
        self.scaler = RobustScaler()
        self.results_comparison = []

#=================================== Preprocessing ===================================#  
    def prepare_rfm_data(self):
        """Preparar datos RFM y normalizarlos"""
        # Preparar DataFrame RFM

        self.df_rfm = self.df[['Nit','02. R Value (Recencia)','03.F value (Frequencia)','02. Contratos Nuevos']].copy()
        self.df_rfm.rename(columns={
            '02. R Value (Recencia)': 'Recencia',
            '03.F value (Frequencia)': 'Frequencia',
            '02. Contratos Nuevos': 'Valor_Monetario'
        }, inplace=True)
        
        # Crear cuartiles para RFM clásico
        self.df_rfm['r_quartile'] = pd.qcut(self.df_rfm['Recencia'], 4, ['1','2','3','4'])
        self.df_rfm['f_quartile'] = pd.qcut(self.df_rfm['Frequencia'], 4, ['4','3','2','1'])
        self.df_rfm['m_quartile'] = pd.qcut(self.df_rfm['Valor_Monetario'], 4, ['4','3','2','1'])
        self.df_rfm['RFMScore'] = (self.df_rfm['r_quartile'].astype(str) + 
                                   self.df_rfm['f_quartile'].astype(str) + 
                                   self.df_rfm['m_quartile'].astype(str))
        
        # Normalizar datos con manejo de outliers
        #numeric_cols = ['Recencia', 'Frequencia', 'Valor_Monetario']
        #self.df_rfm[numeric_cols] = self.df_rfm[numeric_cols].apply(lambda x: np.log1p(x))
        X = self.df_rfm[['Recencia', 'Frequencia', 'Valor_Monetario']]
    
        # Clip outliers para mejorar normalización
        X_winsorized = X.apply(lambda x: winsorize(x, limits=[0.05, 0.05]))
        X_clipped = X_winsorized.clip(lower=X_winsorized.quantile(0.05), upper=X_winsorized.quantile(0.95), axis=1)
        self.X_scaled = self.scaler.fit_transform(X_clipped)

        
        print("✅ Datos RFM preparados y normalizados")
        return self.df_rfm
    
#=================================== Métricas de Evaluación ===================================#   
    def _calculate_metrics(self, labels, X_data, algorithm_name, params=None):
        """Calcular métricas de evaluación para clustering"""
        # Filtrar ruido si existe
        mask = labels != -1
        if mask.sum() < 2:
            return None
            
        X_clean = X_data[mask] if mask.sum() < len(X_data) else X_data
        labels_clean = labels[mask] if mask.sum() < len(labels) else labels
        
        if len(set(labels_clean)) < 2:
            return None
            
        try:
            silhouette = silhouette_score(X_clean, labels_clean)
            dbi = davies_bouldin_score(X_clean, labels_clean)
            chi = calinski_harabasz_score(X_clean, labels_clean)

            # Calcular WCSS (Within-Cluster Sum of Squares)
            wcss = 0
            unique_labels = set(labels_clean)
            for label in unique_labels:
                cluster_points = X_clean[labels_clean == label]
                if len(cluster_points) > 0:
                    cluster_center = cluster_points.mean(axis=0)
                    wcss += ((cluster_points - cluster_center) ** 2).sum()

            result = {
                'Algorithm': algorithm_name,
                'Parameters': str(params) if params else 'Default',
                'N_Clusters': len(set(labels_clean)),
                'N_Noise': sum(labels == -1) if -1 in labels else 0,
                'Silhouette_Score': round(silhouette, 4),
                'Davies_Bouldin': round(dbi, 4),
                'Calinski_Harabasz': round(chi, 4),
                'WCSS': round(wcss, 4),
                'Labels': labels
            }
            return result
        except:
            return None

#=================================== Clustering Methods ===================================#
    def apply_fixed_fuzzy_cmeans_pso(self, c_range=(5, 8), n_particles=15, max_iter=30):
            """Fuzzy C-Means + PSO implementación propia para evitar errores de scikit-fuzzy"""
            print("\U0001F504 Aplicando Fuzzy C-Means + PSO (Implementación Propia)...")

            def custom_fuzzy_cmeans(X, c, m=2.0, max_iter=300, error=0.005):
                """Implementación propia de Fuzzy C-Means"""
                try:
                    n_samples, n_features = X.shape
                    
                    # Inicializar matriz de membresía U aleatoriamente
                    np.random.seed(42)
                    U = np.random.rand(c, n_samples)
                    U = U / np.sum(U, axis=0)  # Normalizar para que sume 1 por columna
                    
                    # Inicializar centros
                    centers = np.zeros((c, n_features))
                    
                    for iteration in range(max_iter):
                        U_old = U.copy()
                        
                        # Calcular centros
                        Um = U ** m
                        for i in range(c):
                            centers[i] = np.sum(Um[i][:, np.newaxis] * X, axis=0) / np.sum(Um[i])
                        
                        # Calcular nuevas membresías
                        for i in range(c):
                            for j in range(n_samples):
                                distances = []
                                for k in range(c):
                                    dist = np.linalg.norm(X[j] - centers[k])
                                    if dist == 0:
                                        dist = 1e-10  # Evitar división por cero
                                    distances.append(dist)
                                
                                total = 0
                                for k in range(c):
                                    if distances[k] > 0:
                                        total += (distances[i] / distances[k]) ** (2 / (m - 1))
                                
                                if total > 0:
                                    U[i, j] = 1.0 / total
                                else:
                                    U[i, j] = 1.0
                        
                        # Normalizar U
                        U = U / np.sum(U, axis=0)
                        
                        # Verificar convergencia
                        if np.max(np.abs(U - U_old)) < error:
                            break
                    
                    # Calcular métricas
                    labels = np.argmax(U, axis=0)
                    
                    # Calcular FPC (Fuzzy Partition Coefficient)
                    fpc = np.sum(U ** 2) / n_samples
                    
                    # Calcular función objetivo
                    jm = 0
                    for i in range(c):
                        for j in range(n_samples):
                            dist = np.linalg.norm(X[j] - centers[i])
                            jm += (U[i, j] ** m) * (dist ** 2)
                    
                    return centers, U, labels, jm, fpc
                    
                except Exception as e:
                    print(f"Error en Fuzzy C-Means personalizado: {e}")
                    return None, None, None, None, None

            def safe_pso_optimize_centers(X, c, n_particles=15, max_iter=30):
                """PSO con manejo seguro de errores"""
                try:
                    n_features = X.shape[1]
                    n_samples = X.shape[0]

                    if c >= n_samples:
                        print(f"\u26A0\ufe0f Clusters ({c}) >= muestras ({n_samples}), usando K-means")
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=min(c, n_samples//2), random_state=42)
                        kmeans.fit(X)
                        return kmeans.cluster_centers_

                    data_min = X.min(axis=0)
                    data_max = X.max(axis=0)
                    data_range = data_max - data_min

                    particles = np.zeros((n_particles, c, n_features))

                    # Inicializar partículas con K-means
                    from sklearn.cluster import KMeans
                    for i in range(n_particles):
                        try:
                            kmeans = KMeans(n_clusters=c, init='k-means++', n_init=1, random_state=i)
                            kmeans.fit(X)
                            particles[i] = kmeans.cluster_centers_
                        except:
                            for j in range(c):
                                particles[i, j] = data_min + np.random.rand(n_features) * data_range

                    velocities = np.random.randn(n_particles, c, n_features) * 0.01 * data_range
                    personal_best = particles.copy()
                    personal_best_scores = np.full(n_particles, float('inf'))
                    global_best = None
                    global_best_score = float('inf')

                    w = 0.7
                    c1 = 1.5
                    c2 = 1.5

                    for iteration in range(max_iter):
                        for i in range(n_particles):
                            centers = particles[i]

                            try:
                                distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                                closest_centers = np.argmin(distances, axis=1)

                                unique_clusters = np.unique(closest_centers)
                                if len(unique_clusters) < c:
                                    fitness = float('inf')
                                else:
                                    fitness = np.sum(np.min(distances, axis=1))

                                    # Penalizar centros muy cercanos
                                    for j in range(c):
                                        for k in range(j+1, c):
                                            dist_centers = np.linalg.norm(centers[j] - centers[k])
                                            if dist_centers < 1e-6:
                                                fitness += 1000
                            except:
                                fitness = float('inf')

                            if fitness < personal_best_scores[i]:
                                personal_best_scores[i] = fitness
                                personal_best[i] = particles[i].copy()

                            if fitness < global_best_score:
                                global_best_score = fitness
                                global_best = particles[i].copy()

                        if global_best is not None:
                            for i in range(n_particles):
                                r1, r2 = np.random.rand(2)
                                velocities[i] = (w * velocities[i] +
                                                c1 * r1 * (personal_best[i] - particles[i]) +
                                                c2 * r2 * (global_best - particles[i]))

                                max_velocity = 0.1 * data_range
                                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

                                particles[i] += velocities[i]
                                particles[i] = np.clip(particles[i], data_min, data_max)

                    return global_best if global_best is not None else particles[0]

                except Exception as e:
                    print(f"Error en PSO: {e}")
                    try:
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=c, random_state=42)
                        kmeans.fit(X)
                        return kmeans.cluster_centers_
                    except:
                        return X[np.random.choice(len(X), c, replace=False)]

            best_result = None
            best_score = -float('inf')

            c_range = (max(2, c_range[0]), min(c_range[1], len(self.X_scaled)//10))
            print(f"  Probando clusters en rango: {c_range[0]} a {c_range[1]-1}")

            for c in range(c_range[0], c_range[1]):
                try:
                    print(f"  Probando {c} clusters...")
                    
                    # Usar PSO para obtener centros iniciales (opcional)
                    optimized_centers = safe_pso_optimize_centers(self.X_scaled, c, n_particles, max_iter)
                    
                    # Aplicar Fuzzy C-Means personalizado
                    centers, U, labels, jm, fpc = custom_fuzzy_cmeans(self.X_scaled, c)
                    
                    if labels is None:
                        print(f"  ❌ Error en FCM personalizado para {c} clusters")
                        continue

                    if len(np.unique(labels)) < 2:
                        print(f"  \u26A0\ufe0f Clusters insuficientes para {c}")
                        continue

                    result = self._calculate_metrics( labels, self.X_scaled,'Fuzzy C-Means + PSO Custom',{'k': c})
                    if result and result['Silhouette_Score'] > best_score:
                        best_score = result['Silhouette_Score']
                    best_result = result

                    if result and result['Silhouette_Score'] > best_score:
                        best_score = result['Silhouette_Score']
                        best_result = result
                        print(f"  ✅ Nuevo mejor resultado para {c} clusters: {best_score:.4f}")

                except Exception as e:
                    print(f"  ❌ Error general para {c} clusters: {e}")
                    continue

            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm['Fuzzy C-Means + PSO'] = best_result['Labels']
                print(f"🏀 Fuzzy C-Means + PSO completado - Mejor Score: {best_result['Silhouette_Score']:.4f}")
            else:
                print("❌ No se pudo completar Fuzzy C-Means + PSO")

            return best_result

    def apply_enhanced_deep_embedded_clustering(self, c_range=(4, 7), encoding_dim=8, epochs=300):
        """Aplicar Deep Embedded Clustering mejorado"""
        print("🔄 Aplicando Deep Embedded Clustering Mejorado...")
        
        def build_enhanced_autoencoder(input_dim, encoding_dim):
            """Construir autoencoder mejorado con regularización"""
            input_layer = keras.Input(shape=(input_dim,))
            
            # Encoder con dropout y batch normalization
            encoded = layers.Dense(64, activation='relu')(input_layer)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(0.2)(encoded)
            
            encoded = layers.Dense(32, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(0.1)(encoded)
            
            encoded = layers.Dense(16, activation='relu')(encoded)
            encoded = layers.Dense(encoding_dim, activation='linear', name='encoded')(encoded)
            
            # Decoder simétrico
            decoded = layers.Dense(16, activation='relu')(encoded)
            decoded = layers.Dense(32, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            
            decoded = layers.Dense(64, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dense(input_dim, activation='linear')(decoded)
            
            autoencoder = keras.Model(input_layer, decoded)
            encoder = keras.Model(input_layer, encoded)
            
            return autoencoder, encoder
        
        def student_t_distribution(z, centers, alpha=1.0):
            """Distribución t-Student mejorada"""
            q = 1.0 / (1.0 + (np.linalg.norm(z[:, np.newaxis] - centers, axis=2) ** 2) / alpha)
            q = q ** ((alpha + 1.0) / 2.0)
            q = (q.T / q.sum(axis=1)).T
            return q
        
        def target_distribution(q):
            """Calcular distribución objetivo mejorada"""
            weight = q ** 2 / q.sum(0)
            return (weight.T / weight.sum(1)).T
        
        best_result = None
        best_score = -1
        
        for c in range(c_range[0], c_range[1]):
            try:
                # Construir autoencoder mejorado
                autoencoder, encoder = build_enhanced_autoencoder(self.X_scaled.shape[1], encoding_dim)
                
                # Compilar con optimizador personalizado
                optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
                autoencoder.compile(optimizer=optimizer, loss='huber')
                
                # Callbacks para entrenamiento
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='loss', patience=20, restore_best_weights=True)
                reduce_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor='loss', factor=0.8, patience=10, min_lr=1e-6)
                
                # Pre-entrenamiento mejorado
                autoencoder.fit(self.X_scaled, self.X_scaled, 
                            epochs=epochs//3, batch_size=min(64, len(self.X_scaled)//4),
                            verbose=0, callbacks=[early_stopping, reduce_lr])
                
                # Obtener representación codificada
                encoded_data = encoder.predict(self.X_scaled, verbose=0)
                
                # Inicialización mejorada con múltiples intentos
                best_init_score = -1
                best_centers = None
                
                for init_attempt in range(5):
                    kmeans = KMeans(n_clusters=c, n_init=20, random_state=42+init_attempt, 
                                init='k-means++', max_iter=500)
                    labels_init = kmeans.fit_predict(encoded_data)
                    init_score = silhouette_score(encoded_data, labels_init)
                    
                    if init_score > best_init_score:
                        best_init_score = init_score
                        best_centers = kmeans.cluster_centers_
                
                # Entrenamiento DEC mejorado
                tolerance = 0.001
                max_epochs = epochs
                prev_assignment = np.zeros(len(self.X_scaled))
                
                for epoch in range(max_epochs):
                    # Calcular distribución Q con t-Student
                    q = student_t_distribution(encoded_data, best_centers)
                    
                    # Calcular distribución P
                    p = target_distribution(q)
                    
                    # Verificar convergencia
                    current_assignment = np.argmax(q, axis=1)
                    delta = np.sum(current_assignment != prev_assignment) / len(current_assignment)
                    
                    if epoch > 0 and delta < tolerance:
                        print(f"Convergencia alcanzada en época {epoch}")
                        break
                    
                    prev_assignment = current_assignment.copy()
                    
                    # Entrenar una época con loss personalizado
                    history = autoencoder.fit(self.X_scaled, self.X_scaled, 
                                            batch_size=min(64, len(self.X_scaled)//4), 
                                            epochs=1, verbose=0)
                    
                    # Actualizar representación y centros
                    encoded_data = encoder.predict(self.X_scaled, verbose=0)
                    for i in range(c):
                        mask = current_assignment == i
                        if np.sum(mask) > 0:
                            best_centers[i] = encoded_data[mask].mean(axis=0)
                
                # Etiquetas finales
                final_q = student_t_distribution(encoded_data, best_centers)
                final_labels = np.argmax(final_q, axis=1)
                
                result = self._calculate_metrics(final_labels, self.X_scaled, 
                                                    'Deep Embedded Clustering Enhanced', 
                                                    {'k': c})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                    
            except Exception as e:
                continue
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['DEC_Enhanced_Cluster'] = best_result['Labels']
            print(f"✅ Deep Embedded Clustering Mejorado completado - Score: {best_result['Silhouette_Score']}")
        return best_result

    def apply_optimized_variational_deep_embedding(self, c_range=(4, 8), latent_dim=8, epochs=300):
        """Aplicar Variational Deep Embedding optimizado"""
        print("🔄 Aplicando Variational Deep Embedding Optimizado...")
        
        def build_optimized_vade_model(input_dim, latent_dim, n_clusters):
            """Construir modelo VaDE optimizado"""
            # Encoder mejorado
            inputs = keras.Input(shape=(input_dim,))
            h = layers.Dense(128, activation='relu')(inputs)
            h = layers.BatchNormalization()(h)
            h = layers.Dropout(0.3)(h)
            
            h = layers.Dense(64, activation='relu')(h)
            h = layers.BatchNormalization()(h)
            h = layers.Dropout(0.2)(h)
            
            h = layers.Dense(32, activation='relu')(h)
            h = layers.Dropout(0.1)(h)
            
            # Parámetros de distribución latente
            z_mean = layers.Dense(latent_dim, name='z_mean')(h)
            z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)
            
            # Sampling layer mejorado
            def sampling(args):
                z_mean, z_log_var = args
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
            
            # Decoder mejorado
            decoder_h = layers.Dense(32, activation='relu')(z)
            decoder_h = layers.Dropout(0.1)(decoder_h)
            decoder_h = layers.Dense(64, activation='relu')(decoder_h)
            decoder_h = layers.BatchNormalization()(decoder_h)
            decoder_h = layers.Dense(128, activation='relu')(decoder_h)
            decoder_h = layers.BatchNormalization()(decoder_h)
            outputs = layers.Dense(input_dim, activation='linear')(decoder_h)
            
            # Clustering layer mejorado
            gamma = layers.Dense(n_clusters, activation='softmax', name='gamma')(z)
            
            # Modelos
            vae = keras.Model(inputs, outputs, name='vae')
            encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
            gamma_model = keras.Model(inputs, gamma, name='gamma_model')
            
            return vae, encoder, gamma_model
        
        class VaDELoss(keras.losses.Loss):
            """Loss personalizado para VaDE"""
            def __init__(self, n_clusters, latent_dim, beta=1.0, **kwargs):
                super().__init__(**kwargs)
                self.n_clusters = n_clusters
                self.latent_dim = latent_dim
                self.beta = beta
            
            def call(self, y_true, y_pred):
                # Pérdida de reconstrucción (Huber loss más robusto)
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.huber(y_true, y_pred, delta=1.0))
                return reconstruction_loss
        
        best_result = None
        best_score = -1
        
        for c in range(c_range[0], c_range[1]):
            try:
                # Construir modelo VaDE optimizado
                vae, encoder, gamma_model = build_optimized_vade_model(
                    self.X_scaled.shape[1], latent_dim, c)
                
                # Optimizador personalizado
                optimizer = keras.optimizers.Adam(
                    learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
                
                # Loss personalizado
                custom_loss = VaDELoss(c, latent_dim)
                vae.compile(optimizer=optimizer, loss=custom_loss)
                
                # Callbacks mejorados
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='loss', patience=25, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='loss', factor=0.7, patience=15, min_lr=1e-6),
                    keras.callbacks.TerminateOnNaN()
                ]
                
                # Pre-entrenamiento como VAE
                vae.fit(self.X_scaled, self.X_scaled, 
                    epochs=epochs//2, batch_size=min(64, len(self.X_scaled)//4),
                    verbose=0, callbacks=callbacks)
                
                # Inicialización con K-means en espacio latente
                z_mean, z_log_var, z_samples = encoder.predict(self.X_scaled, verbose=0)
                kmeans = KMeans(n_clusters=c, n_init=10, random_state=42)
                kmeans_labels = kmeans.fit_predict(z_samples)
                
                # Entrenamiento conjunto mejorado
                batch_size = min(64, len(self.X_scaled)//4)
                n_batches = len(self.X_scaled) // batch_size
                
                for epoch in range(epochs//2):
                    epoch_loss = 0
                    
                    # Entrenamiento por lotes
                    for batch_idx in range(n_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(self.X_scaled))
                        batch_data = self.X_scaled[start_idx:end_idx]
                        
                        # Entrenamiento del lote
                        batch_loss = vae.train_on_batch(batch_data, batch_data)
                        epoch_loss += batch_loss
                    
                    # Early stopping basado en pérdida
                    if epoch > 10 and epoch_loss > epoch_loss * 1.1:
                        break
                
                # Obtener etiquetas finales usando clustering en espacio latente
                final_z_mean, _, final_z = encoder.predict(self.X_scaled, verbose=0)
                final_gamma = gamma_model.predict(self.X_scaled, verbose=0)
                
                # Combinar información de gamma y clustering directo
                gamma_labels = np.argmax(final_gamma, axis=1)
                kmeans_final = KMeans(n_clusters=c, n_init=10, random_state=42)
                direct_labels = kmeans_final.fit_predict(final_z)
                
                # Elegir mejor asignación basada en silhouette
                gamma_score = silhouette_score(self.X_scaled, gamma_labels)
                direct_score = silhouette_score(self.X_scaled, direct_labels)
                
                final_labels = gamma_labels if gamma_score > direct_score else direct_labels
                
                result = self._calculate_metrics(final_labels, self.X_scaled, 'Variational Deep Embedding Optimized', 
                                                    {'k': c})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                    
            except Exception as e:
                continue
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['VaDE_Optimized_Cluster'] = best_result['Labels']
            print(f"✅ Variational Deep Embedding Optimizado completado - Score: {best_result['Silhouette_Score']}")
        return best_result
    def apply_kmeans(self, k_range=(4, 11)):
        """Aplicar K-Means con búsqueda del k óptimo"""
        print("🔄 Aplicando K-Means...")
        
        best_result = None
        best_score = -1
        
        for k in range(k_range[0], k_range[1]):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            result = self._calculate_metrics(labels, self.X_scaled, 'K-Means', {'k': k})
            if result and result['Silhouette_Score'] > best_score:
                best_score = result['Silhouette_Score']
                best_result = result
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['KMeans_Cluster'] = best_result['Labels']
            print(f"✅ K-Means completado - Mejor k: {best_result['Parameters']}")
        return best_result


#===================================== Graph Neural Networks ===================================#
    def apply_gnn_clustering(self, k_range=(5, 11), k_neighbors=8, epochs=100, lr=0.01):
        """
        Aplicar Graph Neural Networks para clustering de datos RFM
        Implementa GCN con clustering sinérgico profundo
        """
        print("🔄 Aplicando GNN Clustering...")
        
        try:
            # Importar librerías necesarias
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from sklearn.neighbors import kneighbors_graph
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Crear grafo k-NN basado en similitud RFM
            print(f"   📊 Construyendo grafo k-NN (k={k_neighbors})...")
            knn_graph = kneighbors_graph(
                self.X_scaled, 
                n_neighbors=k_neighbors, 
                mode='connectivity',
                include_self=False
            )
            
            # Convertir a formato de edges
            edges = np.array(knn_graph.nonzero())
            edge_index = torch.tensor(edges, dtype=torch.long)
            node_features = torch.tensor(self.X_scaled, dtype=torch.float32)
            
            # Definir modelo GNN mejorado
            class RFM_GCN(nn.Module):
                def __init__(self, input_dim, hidden_dim, embedding_dim):
                    super(RFM_GCN, self).__init__()
                    self.input_dim = input_dim
                    self.hidden_dim = hidden_dim
                    self.embedding_dim = embedding_dim
                    
                    # Capas de transformación
                    self.conv1 = nn.Linear(input_dim, hidden_dim)
                    self.conv2 = nn.Linear(hidden_dim, embedding_dim)
                    self.reconstruction = nn.Linear(embedding_dim, input_dim)
                    
                    self.dropout = nn.Dropout(0.1)
                    self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                    self.batch_norm2 = nn.BatchNorm1d(embedding_dim)
                    
                def forward(self, x, edge_index):
                    # Primera capa con agregación de vecinos
                    x1 = self.message_passing(x, edge_index, self.conv1)
                    x1 = self.batch_norm1(x1)
                    x1 = F.relu(x1)
                    x1 = self.dropout(x1)
                    
                    # Segunda capa (embeddings)
                    embeddings = self.message_passing(x1, edge_index, self.conv2)
                    embeddings = self.batch_norm2(embeddings)
                    embeddings = F.relu(embeddings)
                    
                    # Normalización L2 para clustering
                    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                    
                    return embeddings_norm, embeddings
                
                def message_passing(self, x, edge_index, linear_layer):
                    """Agregación simple de vecinos con self-attention"""
                    row, col = edge_index
                    
                    # Aplicar transformación lineal
                    x_transformed = linear_layer(x)
                    
                    # Agregación mejorada con peso de auto-conexión
                    out = torch.zeros_like(x_transformed)
                    
                    for i in range(x.size(0)):
                        neighbors = col[row == i]
                        if len(neighbors) > 0:
                            # Auto-conexión + vecinos con pesos
                            self_feature = x_transformed[i:i+1]
                            neighbor_features = x_transformed[neighbors]
                            
                            # Promedio ponderado (más peso a sí mismo)
                            all_features = torch.cat([self_feature * 0.5, neighbor_features * 0.5 / len(neighbors)], dim=0)
                            out[i] = torch.sum(all_features, dim=0)
                        else:
                            out[i] = x_transformed[i]
                    
                    return out
                
                def reconstruct(self, embeddings):
                    """Reconstruir features originales"""
                    return self.reconstruction(embeddings)
            
            best_result = None
            best_score = -1
            
            print("   🧠 Entrenando modelos GNN para diferentes valores de k...")
            
            for k in range(k_range[0], k_range[1]):
                try:
                    # Crear modelo
                    model = RFM_GCN(
                        input_dim=node_features.shape[1],  # 3 para RFM
                        hidden_dim=16,
                        embedding_dim=8
                    )
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                    
                    # Variables para clustering
                    cluster_centers = None
                    
                    # Entrenamiento
                    model.train()
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        
                        # Forward pass
                        embeddings_norm, embeddings_raw = model(node_features, edge_index)
                        
                        # Reconstrucción
                        reconstructed = model.reconstruct(embeddings_raw)
                        reconstruction_loss = F.mse_loss(reconstructed, node_features)
                        
                        # Clustering loss cada 20 épocas
                        clustering_loss = torch.tensor(0.0)
                        if epoch % 20 == 0 and epoch > 0:
                            with torch.no_grad():
                                # Usar K-means para obtener centros
                                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=3)
                                temp_labels = kmeans_temp.fit_predict(embeddings_norm.numpy())
                                
                                # Calcular centros en el espacio de embeddings
                                cluster_centers = torch.zeros(k, embeddings_norm.shape[1])
                                for i in range(k):
                                    mask = temp_labels == i
                                    if mask.sum() > 0:
                                        cluster_centers[i] = embeddings_norm[mask].mean(dim=0)
                        
                        # Aplicar clustering loss si hay centros
                        if cluster_centers is not None:
                            # Distancia a centros más cercanos
                            distances = torch.cdist(embeddings_norm, cluster_centers)
                            min_distances, assigned_clusters = torch.min(distances, dim=1)
                            clustering_loss = torch.mean(min_distances)
                        
                        # Loss total
                        total_loss = reconstruction_loss + 0.1 * clustering_loss
                        
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # Log cada 50 épocas
                        if epoch % 50 == 0:
                            print(f"      Época {epoch}: Loss={total_loss.item():.4f} (Recon: {reconstruction_loss.item():.4f}, Cluster: {clustering_loss.item():.4f})")
                    
                    # Obtener embeddings finales
                    model.eval()
                    with torch.no_grad():
                        final_embeddings, _ = model(node_features, edge_index)
                        final_embeddings = final_embeddings.numpy()
                    
                    # Clustering final con múltiples intentos
                    best_labels = None
                    best_silhouette = -1
                    
                    for attempt in range(5):
                        kmeans_final = KMeans(n_clusters=k, random_state=42+attempt, n_init=10)
                        labels_attempt = kmeans_final.fit_predict(final_embeddings)
                        
                        # Verificar calidad
                        if len(set(labels_attempt)) == k:  # Todos los clusters representados
                            try:
                                from sklearn.metrics import silhouette_score
                                sil_score = silhouette_score(final_embeddings, labels_attempt)
                                if sil_score > best_silhouette:
                                    best_silhouette = sil_score
                                    best_labels = labels_attempt
                            except:
                                continue
                    
                    if best_labels is None:
                        best_labels = labels_attempt  # Usar último intento si no hay mejor
                    
                    # Calcular métricas
                    result = self._calculate_metrics(
                        best_labels, 
                        final_embeddings, 
                        'GNN', 
                        {
                            'k': k, 
                            'k_neighbors': k_neighbors,
                            'epochs': epochs,
                            'embedding_dim': final_embeddings.shape[1],
                            'hidden_dim': 16
                        }
                    )
                    
                    if result and result['Silhouette_Score'] > best_score:
                        best_score = result['Silhouette_Score']
                        best_result = result
                        
                    print(f"   📈 k={k}: Score={result['Silhouette_Score']:.4f}" if result else f"   ❌ k={k}: Error en métricas")
                    
                except Exception as e:
                    print(f"   ⚠️  Error con k={k}: {str(e)}")
                    continue
            
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm['GNN_Cluster'] = best_result['Labels']
                print(f"✅ GNN completado - Mejor configuración: {best_result['Parameters']}")
                print(f"   🎯 Mejor Silhouette Score: {best_result['Silhouette_Score']:.4f}")
                print(f"   📊 Clusters encontrados: {best_result['N_Clusters']}")
            else:
                print("❌ No se pudo completar el clustering GNN")
                
            return best_result
            
        except ImportError as e:
            print("❌ Error: Librerías de PyTorch no disponibles")
            print("   💡 Instalación requerida: pip install torch")
            return None
            
        except Exception as e:
            print(f"❌ Error en GNN Clustering: {str(e)}")
            return None

    def apply_gnn_guided_hdbscan(self, k_neighbors=8, epochs=100, lr=0.01, min_size_range=(5, 31, 5)):
        """
        Aplicar clustering híbrido: GNN para embeddings mejorados + HDBSCAN para clustering
        Combina la representación estructural de GNN con la robustez de HDBSCAN
        """
        print("🔄 Aplicando GNN-Guided HDBSCAN...")
        
        try:
            # Importar librerías necesarias
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from sklearn.neighbors import kneighbors_graph
            import numpy as np
            import hdbscan
            
            # FASE 1: Generar embeddings con GNN
            print(f"   🧠 Fase 1: Generando embeddings con GNN (k_neighbors={k_neighbors})...")
            
            # Crear grafo k-NN basado en similitud RFM
            knn_graph = kneighbors_graph(
                self.X_scaled, 
                n_neighbors=k_neighbors, 
                mode='connectivity',
                include_self=False
            )
            
            # Convertir a formato de edges
            edges = np.array(knn_graph.nonzero())
            edge_index = torch.tensor(edges, dtype=torch.long)
            node_features = torch.tensor(self.X_scaled, dtype=torch.float32)
            
            # Definir modelo GNN optimizado para embeddings
            class EmbeddingGCN(nn.Module):
                def __init__(self, input_dim, hidden_dim, embedding_dim):
                    super(EmbeddingGCN, self).__init__()
                    self.input_dim = input_dim
                    self.hidden_dim = hidden_dim
                    self.embedding_dim = embedding_dim
                    
                    # Capas de transformación
                    self.conv1 = nn.Linear(input_dim, hidden_dim)
                    self.conv2 = nn.Linear(hidden_dim, embedding_dim)
                    self.conv3 = nn.Linear(embedding_dim, embedding_dim)  # Capa adicional para refinamiento
                    
                    self.dropout = nn.Dropout(0.15)
                    self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                    self.batch_norm2 = nn.BatchNorm1d(embedding_dim)
                    self.batch_norm3 = nn.BatchNorm1d(embedding_dim)
                    
                def forward(self, x, edge_index):
                    # Primera capa con agregación de vecinos
                    x1 = self.message_passing(x, edge_index, self.conv1)
                    x1 = self.batch_norm1(x1)
                    x1 = F.leaky_relu(x1, 0.2)
                    x1 = self.dropout(x1)
                    
                    # Segunda capa (embeddings intermedios)
                    x2 = self.message_passing(x1, edge_index, self.conv2)
                    x2 = self.batch_norm2(x2)
                    x2 = F.leaky_relu(x2, 0.2)
                    
                    # Tercera capa (embeddings finales)
                    embeddings = self.message_passing(x2, edge_index, self.conv3)
                    embeddings = self.batch_norm3(embeddings)
                    
                    # Normalización L2 para clustering
                    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                    
                    return embeddings_norm, embeddings
                
                def message_passing(self, x, edge_index, linear_layer):
                    """Agregación mejorada con atención basada en similitud"""
                    row, col = edge_index
                    x_transformed = linear_layer(x)
                    out = torch.zeros_like(x_transformed)
                    
                    for i in range(x.size(0)):
                        neighbors = col[row == i]
                        if len(neighbors) > 0:
                            # Feature del nodo actual
                            self_feature = x_transformed[i:i+1]
                            neighbor_features = x_transformed[neighbors]
                            
                            # Calcular similitudes como pesos de atención
                            similarities = F.cosine_similarity(
                                self_feature.repeat(len(neighbors), 1), 
                                neighbor_features,
                                dim=1
                            )
                            weights = F.softmax(similarities, dim=0)
                            
                            # Agregación ponderada con mayor peso propio
                            weighted_neighbors = torch.sum(
                                neighbor_features * weights.unsqueeze(1), dim=0
                            )
                            out[i] = 0.6 * x_transformed[i] + 0.4 * weighted_neighbors
                        else:
                            out[i] = x_transformed[i]
                    
                    return out
            
            # Buscar mejores embeddings probando diferentes dimensiones
            best_embeddings = None
            best_embedding_score = -1
            best_embedding_params = None
            
            print("   🔍 Optimizando dimensión de embeddings...")
            
            for embedding_dim in [8, 12, 16, 20]:
                try:
                    # Crear modelo
                    model = EmbeddingGCN(
                        input_dim=node_features.shape[1],  # 3 para RFM
                        hidden_dim=24,
                        embedding_dim=embedding_dim
                    )
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
                    
                    # Entrenar modelo
                    model.train()
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        
                        # Forward pass
                        embeddings_norm, embeddings_raw = model(node_features, edge_index)
                        
                        # Loss combinado: estructura local + separación global
                        local_loss = self._compute_local_structure_loss(embeddings_norm, edge_index)
                        separation_loss = self._compute_cluster_separation_loss(embeddings_norm, k=6)
                        
                        total_loss = local_loss + 0.3 * separation_loss
                        
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        
                        # Log progreso
                        if epoch % 30 == 0:
                            print(f"      Dim {embedding_dim}, Época {epoch}: Loss={total_loss.item():.4f}")
                    
                    # Obtener embeddings finales
                    model.eval()
                    with torch.no_grad():
                        test_embeddings, _ = model(node_features, edge_index)
                        test_embeddings = test_embeddings.numpy()
                    
                    # Evaluación rápida con HDBSCAN
                    test_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
                    test_labels = test_clusterer.fit_predict(test_embeddings)
                    
                    # Verificar calidad
                    n_clusters_test = len(set(test_labels)) - (1 if -1 in test_labels else 0)
                    if n_clusters_test >= 2:
                        try:
                            from sklearn.metrics import silhouette_score
                            test_score = silhouette_score(test_embeddings, test_labels)
                            if test_score > best_embedding_score:
                                best_embedding_score = test_score
                                best_embeddings = test_embeddings
                                best_embedding_params = {
                                    'embedding_dim': embedding_dim,
                                    'hidden_dim': 24,
                                    'test_score': test_score,
                                    'test_clusters': n_clusters_test
                                }
                                print(f"      ✨ Mejor configuración: dim={embedding_dim}, score={test_score:.4f}, clusters={n_clusters_test}")
                        except:
                            continue
                    
                except Exception as e:
                    print(f"      ⚠️  Error con embedding_dim={embedding_dim}: {str(e)}")
                    continue
            
            if best_embeddings is None:
                print("❌ No se pudieron generar embeddings válidos")
                return None
            
            print(f"   🎯 Mejores embeddings: {best_embedding_params}")
            
            # FASE 2: Aplicar HDBSCAN a los embeddings optimizados
            print(f"   🔍 Fase 2: Aplicando HDBSCAN a embeddings GNN...")
            
            best_result = None
            best_score = -1
            
            for min_size in range(min_size_range[0], min_size_range[1], min_size_range[2]):
                for min_samples in [3, 5, 7]:
                    try:
                        # Crear clusterer HDBSCAN
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_size,
                            min_samples=min_samples,
                            cluster_selection_method='leaf',
                            metric='euclidean'
                        )
                        
                        labels = clusterer.fit_predict(best_embeddings)
                        
                        # Verificar calidad del clustering
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        noise_ratio = n_noise / len(labels)
                        
                        # Criterios de calidad
                        if n_clusters >= 3 and noise_ratio < 0.4:
                            result = self._calculate_metrics(
                                labels, 
                                best_embeddings, 
                                'GNN-HDBSCAN', 
                                {
                                    'k_neighbors': k_neighbors,
                                    'epochs': epochs,
                                    'embedding_dim': best_embedding_params['embedding_dim'],
                                    'hidden_dim': best_embedding_params['hidden_dim'],
                                    'min_cluster_size': min_size,
                                    'min_samples': min_samples,
                                    'n_noise_points': n_noise,
                                    'noise_ratio': noise_ratio,
                                    'embedding_quality_score': best_embedding_score
                                }
                            )
                            
                            if result and result['Silhouette_Score'] > best_score:
                                best_score = result['Silhouette_Score']
                                best_result = result
                                print(f"      📈 min_size={min_size}, min_samples={min_samples}: "
                                    f"Score={best_score:.4f}, Clusters={n_clusters}, Noise={n_noise}")
                    
                    except Exception as e:
                        print(f"      ⚠️  Error con min_size={min_size}, min_samples={min_samples}: {str(e)}")
                        continue
            
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm['GNN_HDBSCAN_Cluster'] = best_result['Labels']
                print(f"✅ GNN-HDBSCAN Híbrido completado")
                print(f"   🎯 Mejor Silhouette Score: {best_result['Silhouette_Score']:.4f}")
                print(f"   📊 Clusters encontrados: {best_result['N_Clusters']}")
                print(f"   🔇 Puntos de ruido: {best_result['Parameters']['n_noise_points']} ({best_result['Parameters']['noise_ratio']:.1%})")
                print(f"   ⚙️  Configuración: {best_result['Parameters']}")
            else:
                print("❌ No se pudo completar el clustering GNN-HDBSCAN")
                
            return best_result
            
        except ImportError as e:
            print("❌ Error: Librerías requeridas no disponibles")
            print("   💡 Instalación requerida: pip install torch hdbscan")
            return None
            
        except Exception as e:
            print(f"❌ Error en GNN-HDBSCAN: {str(e)}")
            return None

    def _compute_local_structure_loss(self, embeddings, edge_index):
        """Calcular loss para preservar estructura local del grafo"""
        row, col = edge_index
        
        # Distancias entre nodos conectados (deben ser pequeñas)
        edge_distances = torch.norm(embeddings[row] - embeddings[col], dim=1)
        local_loss = torch.mean(edge_distances)
        
        return local_loss

    def _compute_cluster_separation_loss(self, embeddings, k=6):
        """Calcular loss para fomentar separación entre clusters"""
        try:
            from sklearn.cluster import KMeans
            
            with torch.no_grad():
                # K-means temporal para identificar posibles centros
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
                temp_labels = kmeans.fit_predict(embeddings.detach().numpy())
                
                # Calcular centros de clusters
                centers = torch.zeros(k, embeddings.shape[1])
                for i in range(k):
                    mask = temp_labels == i
                    if mask.sum() > 0:
                        centers[i] = embeddings[mask].mean(dim=0)
            
            # Distancias entre centros (queremos que sean grandes)
            center_distances = torch.cdist(centers, centers)
            
            # Penalizar centros muy cercanos
            mask = torch.eye(k, dtype=torch.bool)
            center_distances = center_distances.masked_fill(mask, float('inf'))
            min_distances = torch.min(center_distances, dim=1)[0]
            
            # Maximizar la distancia mínima entre centros
            separation_loss = -torch.mean(min_distances)
            
            return separation_loss
            
        except Exception as e:
            return torch.tensor(0.0)
#=================================== gnnn con aprendizaje =============================================

#========================================== Rest of all code ==========================================

    def apply_graph_clustering_alternative(self, k_range=(4, 11), method='spectral'):
        """
        Alternativa de clustering basado en grafos sin PyTorch
        Usa clustering espectral y propagación de etiquetas
        """
        print("🔄 Aplicando Graph Clustering (Alternativo)...")
        
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.neighbors import kneighbors_graph
            import networkx as nx
            from sklearn.semi_supervised import LabelPropagation
            
            # Crear grafo k-NN
            knn_graph = kneighbors_graph(
                self.X_scaled, 
                n_neighbors=8, 
                mode='connectivity',
                include_self=False
            )
            
            best_result = None
            best_score = -1
            
            for k in range(k_range[0], k_range[1]):
                try:
                    if method == 'spectral':
                        # Clustering Espectral
                        spectral = SpectralClustering(
                            n_clusters=k,
                            affinity='precomputed',
                            random_state=42,
                            assign_labels='discretize'
                        )
                        labels = spectral.fit_predict(knn_graph)
                        
                    elif method == 'label_propagation':
                        # Propagación de etiquetas semi-supervisada
                        from sklearn.cluster import KMeans
                        kmeans_seed = KMeans(n_clusters=k, random_state=42, n_init=5)
                        seed_labels = kmeans_seed.fit_predict(self.X_scaled)
                        
                        # Seleccionar algunos puntos como semillas
                        labeled_indices = np.random.choice(len(seed_labels), size=k*2, replace=False)
                        y_labeled = np.full(len(seed_labels), -1)
                        y_labeled[labeled_indices] = seed_labels[labeled_indices]
                        
                        lp = LabelPropagation(kernel='knn', n_neighbors=8)
                        labels = lp.fit_predict(self.X_scaled, y_labeled)
                    
                    result = self._calculate_metrics(
                        labels, 
                        self.X_scaled, 
                        f'Graph-{method.title()}', 
                        {'k': k, 'method': method}
                    )
                    
                    if result and result['Silhouette_Score'] > best_score:
                        best_score = result['Silhouette_Score']
                        best_result = result
                        
                except Exception as e:
                    print(f"   ⚠️  Error con k={k}: {str(e)}")
                    continue
            
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm[f'Graph_{method.title()}_Cluster'] = best_result['Labels']
                print(f"✅ Graph Clustering completado - Mejor k: {best_result['Parameters']}")
            
            return best_result
            
        except Exception as e:
            print(f"❌ Error en Graph Clustering: {str(e)}")
            return None
    
    def apply_graph_clustering_alternative(self, k_range=(4, 11), method='spectral'):
        """
        Alternativa de clustering basado en grafos sin PyTorch
        Usa clustering espectral y propagación de etiquetas
        """
        print("🔄 Aplicando Graph Clustering (Alternativo)...")
        
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.neighbors import kneighbors_graph
            import networkx as nx
            from sklearn.semi_supervised import LabelPropagation
            
            # Crear grafo k-NN
            knn_graph = kneighbors_graph(
                self.X_scaled, 
                n_neighbors=8, 
                mode='connectivity',
                include_self=False
            )
            
            best_result = None
            best_score = -1
            
            for k in range(k_range[0], k_range[1]):
                try:
                    if method == 'spectral':
                        # Clustering Espectral
                        spectral = SpectralClustering(
                            n_clusters=k,
                            affinity='precomputed',
                            random_state=42,
                            assign_labels='discretize'
                        )
                        labels = spectral.fit_predict(knn_graph)
                        
                    elif method == 'label_propagation':
                        # Propagación de etiquetas semi-supervisada
                        # Usar algunos puntos como semillas
                        from sklearn.cluster import KMeans
                        kmeans_seed = KMeans(n_clusters=k, random_state=42, n_init=5)
                        seed_labels = kmeans_seed.fit_predict(self.X_scaled)
                        
                        # Seleccionar algunos puntos como etiquetados
                        labeled_indices = np.random.choice(len(seed_labels), size=k*2, replace=False)
                        y_labeled = np.full(len(seed_labels), -1)
                        y_labeled[labeled_indices] = seed_labels[labeled_indices]
                        
                        lp = LabelPropagation(kernel='knn', n_neighbors=8)
                        labels = lp.fit_predict(self.X_scaled, y_labeled)
                    
                    result = self._calculate_metrics(
                        labels, 
                        self.X_scaled, 
                        f'Graph-{method.title()}', 
                        {'k': k, 'method': method}
                    )
                    
                    if result and result['Silhouette_Score'] > best_score:
                        best_score = result['Silhouette_Score']
                        best_result = result
                        
                except Exception as e:
                    print(f"   ⚠️  Error con k={k}: {str(e)}")
                    continue
            
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm[f'Graph_{method.title()}_Cluster'] = best_result['Labels']
                print(f"✅ Graph Clustering completado - Mejor k: {best_result['Parameters']}")
            
            return best_result
            
        except Exception as e:
            print(f"❌ Error en Graph Clustering: {str(e)}")
            return None

    def apply_hybrid_firefly_clustering(self, k_range=(4, 11), n_fireflies=30, max_iter=70):
        """
        Aplicar clustering híbrido con algoritmo Firefly simplificado y robusto
        """
        print("🔄 Aplicando Hybrid Firefly Clustering...")
        
        def safe_silhouette_score(X, labels):
            """Cálculo seguro del silhouette score"""
            try:
                # Asegurar tipos correctos
                X = np.asarray(X, dtype=np.float64)
                labels = np.asarray(labels, dtype=np.int32)
                
                # Validaciones básicas
                if len(X) != len(labels):
                    return -1
                if len(np.unique(labels)) < 2:
                    return -1
                if len(X) < 2:
                    return -1
                    
                return silhouette_score(X, labels)
            except Exception as e:
                print(f"Error en silhouette: {e}")
                return -1
        
        def calculate_fitness(centroids, X):
            """Fitness simplificado basado en inercia y silhouette"""
            try:
                # Validar entrada
                centroids = np.asarray(centroids, dtype=np.float64)
                X = np.asarray(X, dtype=np.float64)
                
                if centroids.shape[1] != X.shape[1]:
                    return 0.001
                    
                # Calcular distancias y asignar clusters
                distances = []
                for point in X:
                    point_distances = [np.sqrt(np.sum((point - centroid)**2)) for centroid in centroids]
                    distances.append(point_distances)
                
                distances = np.array(distances)
                labels = np.argmin(distances, axis=1)
                
                # Verificar que tenemos clusters válidos
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    return 0.001
                
                # Calcular inercia (WCSS)
                wcss = 0
                for i, label in enumerate(unique_labels):
                    cluster_points = X[labels == label]
                    if len(cluster_points) > 0 and label < len(centroids):
                        centroid = centroids[label]
                        cluster_wcss = np.sum((cluster_points - centroid)**2)
                        wcss += cluster_wcss
                
                # Silhouette score como medida adicional
                sil_score = safe_silhouette_score(X, labels)
                
                # Fitness combinado (invertir WCSS y combinar con silhouette)
                fitness = (1 / (1 + wcss)) * 0.7 + max(0, sil_score + 1) * 0.3
                
                return fitness
                
            except Exception as e:
                print(f"Error en fitness: {e}")
                return 0.001
        
        def move_firefly_simple(firefly_i, firefly_j, X_bounds, alpha=0.2, beta=1.0, gamma=0.1):
            """Movimiento simplificado de luciérnaga"""
            try:
                # Calcular distancia
                distance = np.sqrt(np.sum((firefly_i - firefly_j)**2))
                
                # Atracción
                attraction = beta * np.exp(-gamma * distance**2)
                
                # Nuevo movimiento
                new_firefly = firefly_i + attraction * (firefly_j - firefly_i) + \
                            alpha * (np.random.random(firefly_i.shape) - 0.5)
                
                # Aplicar límites
                for i in range(new_firefly.shape[0]):
                    for j in range(new_firefly.shape[1]):
                        new_firefly[i, j] = np.clip(new_firefly[i, j], X_bounds[j][0], X_bounds[j][1])
                
                return new_firefly.astype(np.float64)
                
            except Exception as e:
                print(f"Error en movimiento: {e}")
                return firefly_i
        
        def firefly_optimization(X, k):
            """Optimización Firefly simplificada"""
            try:
                X = np.asarray(X, dtype=np.float64)
                n_features = X.shape[1]
                
                # Calcular límites
                X_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(n_features)]
                
                # Inicializar luciérnagas
                fireflies = []
                for _ in range(n_fireflies):
                    centroids = np.random.uniform(
                        low=[bound[0] for bound in X_bounds],
                        high=[bound[1] for bound in X_bounds],
                        size=(k, n_features)
                    ).astype(np.float64)
                    fireflies.append(centroids)
                
                # Calcular fitness inicial
                brightness = []
                for firefly in fireflies:
                    fit = calculate_fitness(firefly, X)
                    brightness.append(fit)
                
                best_idx = np.argmax(brightness)
                best_firefly = fireflies[best_idx].copy()
                best_brightness = brightness[best_idx]
                
                # Evolución
                for iteration in range(max_iter):
                    for i in range(n_fireflies):
                        for j in range(n_fireflies):
                            if brightness[j] > brightness[i]:
                                # Mover luciérnaga
                                new_firefly = move_firefly_simple(
                                    fireflies[i], fireflies[j], X_bounds
                                )
                                
                                # Evaluar nuevo fitness
                                new_brightness = calculate_fitness(new_firefly, X)
                                
                                # Aceptar si mejora
                                if new_brightness > brightness[i]:
                                    fireflies[i] = new_firefly
                                    brightness[i] = new_brightness
                                    
                                    # Actualizar mejor
                                    if new_brightness > best_brightness:
                                        best_brightness = new_brightness
                                        best_firefly = new_firefly.copy()
                
                return best_firefly
                
            except Exception as e:
                print(f"Error en optimización: {e}")
                # Fallback a centroides aleatorios
                return np.random.uniform(
                    low=[X[:, i].min() for i in range(X.shape[1])],
                    high=[X[:, i].max() for i in range(X.shape[1])],
                    size=(k, X.shape[1])
                ).astype(np.float64)
        
        # Buscar mejor k
        best_result = None
        best_score = -1
        
        for k in range(k_range[0], k_range[1]):
            try:
                print(f"  Probando k={k}...")
                
                # Optimizar centroides
                optimized_centroids = firefly_optimization(self.X_scaled, k)
                
                # Clustering final con K-means
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, init=optimized_centroids, n_init=1, random_state=42)
                labels = kmeans.fit_predict(self.X_scaled)
                
                # Calcular métricas
                params = {'k': k, 'n_fireflies': n_fireflies, 'max_iter': max_iter}
                result = self._calculate_metrics(labels, self.X_scaled, 'Firefly_Hybrid', params)
                
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                    print(f"    ✓ Nuevo mejor: k={k}, score={best_score:.4f}")
                    
            except Exception as e:
                print(f"⚠️ Error en k={k}: {str(e)}")
                continue
        
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['Firefly_Cluster'] = best_result['Labels']
            
            # Extraer k del string de parámetros
            params_str = best_result['Parameters']
            try:
                import re
                k_match = re.search(r"'k':\s*(\d+)", params_str)
                best_k = int(k_match.group(1)) if k_match else "unknown"
            except:
                best_k = "unknown"
                
            print(f"✅ Firefly Clustering completado - Mejor k: {best_k}, Score: {best_result['Silhouette_Score']:.4f}")
        else:
            print("❌ Firefly Clustering falló completamente")
            
        return best_result

    def apply_fuzzy_cmeans(self, c_range=(4, 15)):
        """Aplicar Fuzzy C-Means"""
        print("🔄 Aplicando Fuzzy C-Means...")
        
        X_T = self.X_scaled.T
        best_result = None
        best_score = -1
        
        for c in range(c_range[0], c_range[1]):
            try:
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    X_T, c=c, m=2.0, error=0.005, maxiter=1000, init=None
                )
                labels = np.argmax(u, axis=0)
                
                result = self._calculate_metrics(labels, self.X_scaled, 'Fuzzy C-Means', 
                                               {'c': c, 'fpc': round(fpc, 4)})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
            except:
                continue
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['FuzzyCMeans_Cluster'] = best_result['Labels']
            print(f"✅ Fuzzy C-Means completado")
        return best_result
    
    def apply_dbscan(self, eps_range=(0.1, 2.0, 0.1), min_samples_range=(3, 11)):
        """Aplicar DBSCAN con búsqueda de parámetros"""
        print("🔄 Aplicando DBSCAN...")
        
        eps_values = np.arange(eps_range[0], eps_range[1], eps_range[2])
        min_samples_values = range(min_samples_range[0], min_samples_range[1])
        
        best_result = None
        best_score = -1
        
        for eps in tqdm(eps_values, desc="DBSCAN eps"):
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.X_scaled)
                
                result = self._calculate_metrics(labels, self.X_scaled, 'DBSCAN', 
                                               {'eps': eps, 'min_samples': min_samples})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                    
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['DBSCAN_Cluster'] = best_result['Labels']
            print(f"✅ DBSCAN completado")
        return best_result
    
    def apply_optics(self, min_samples_list=[4, 5, 6, 7], xi_list=[0.005, 0.05, 0.1]):
        """Aplicar OPTICS"""
        print("🔄 Aplicando OPTICS...")
        
        best_result = None
        best_score = -1
        
        for min_samples, xi in product(min_samples_list, xi_list):
            optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=0.05)
            labels = optics.fit_predict(self.X_scaled)
            
            result = self._calculate_metrics(labels, self.X_scaled, 'OPTICS', 
                                           {'min_samples': min_samples, 'xi': xi})
            if result and result['Silhouette_Score'] > best_score:
                best_score = result['Silhouette_Score']
                best_result = result
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['OPTICS_Cluster'] = best_result['Labels']
            print(f"✅ OPTICS completado")
        return best_result
    
    def apply_hdbscan(self, min_size_range=(5, 51, 5)):
        """Aplicar HDBSCAN"""
        print("🔄 Aplicando HDBSCAN...")
        
        best_result = None
        best_score = -1
        
        for min_size in range(min_size_range[0], min_size_range[1], min_size_range[2]):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=5,
                                       cluster_selection_method='leaf')
            labels = clusterer.fit_predict(self.X_scaled)
            
            result = self._calculate_metrics(labels, self.X_scaled, 'HDBSCAN', 
                                           {'min_cluster_size': min_size})
            if result and result['Silhouette_Score'] > best_score:
                best_score = result['Silhouette_Score']
                best_result = result
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['HDBSCAN_Cluster'] = best_result['Labels']
            print(f"✅ HDBSCAN completado")
        return best_result
    
    def apply_meanshift(self, bandwidth_values=[0.1, 1.0, 2.0]):
        """Aplicar Mean Shift"""
        print("🔄 Aplicando Mean Shift...")
        
        best_result = None
        best_score = -1
        
        for bw in bandwidth_values:
            ms = MeanShift(bandwidth=bw, bin_seeding=True)
            labels = ms.fit_predict(self.X_scaled)
            
            result = self._calculate_metrics(labels, self.X_scaled, 'Mean Shift', 
                                           {'bandwidth': bw})
            if result and result['Silhouette_Score'] > best_score:
                best_score = result['Silhouette_Score']
                best_result = result
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['MeanShift_Cluster'] = best_result['Labels']
            print(f"✅ Mean Shift completado")
        return best_result
    
    def apply_gaussian_mixture(self, k_range=(4, 15)):
        """Aplicar Gaussian Mixture Model"""
        print("🔄 Aplicando Gaussian Mixture Model...")
        
        best_result = None
        best_score = -1
        
        for k in range(k_range[0], k_range[1]):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            labels = gmm.fit_predict(self.X_scaled)
            
            result = self._calculate_metrics(labels, self.X_scaled, 'Gaussian Mixture', 
                                           {'n_components': k})
            if result and result['Silhouette_Score'] > best_score:
                best_score = result['Silhouette_Score']
                best_result = result
                
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['GMM_Cluster'] = best_result['Labels']
            print(f"✅ Gaussian Mixture completado")
        return best_result
    
    def apply_spectral_clustering(self, k_range=(4, 8)):
        """Aplicar Spectral Clustering"""
        print("🔄 Aplicando Spectral Clustering...")
    
        best_result = None
        best_score = -1
    
        for k in range(k_range[0], k_range[1]):
            try:
                spectral = SpectralClustering(n_clusters=k, affinity='rbf', 
                                        gamma=1.0, random_state=42, n_jobs=-1)
                labels = spectral.fit_predict(self.X_scaled)
            
                result = self._calculate_metrics(labels, self.X_scaled, 'Spectral Clustering',
                                           {'n_clusters': k, 'affinity': 'rbf'})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
            except Exception as e:
                print(f"⚠️ Error con k={k}: {str(e)}")
                continue
    
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['Spectral_Cluster'] = best_result['Labels']
            print(f"✅ Spectral Clustering completado")
        return best_result

    def apply_birch_clustering(self, n_clusters_range=(4, 8)):
        """Aplicar BIRCH Clustering (eficiente para datasets grandes)"""
        print("🔄 Aplicando BIRCH Clustering...")
        print(f"📊 Shape de datos: {self.X_scaled.shape}")
        
        # Verificar que los datos no tengan NaN o infinitos
        if hasattr(self.X_scaled, 'isna'):
            if self.X_scaled.isna().any().any():
                print("⚠️ Se encontraron valores NaN en los datos")
        
        best_result = None
        best_score = -1
        best_labels = None
        best_model = None

        for k in range(n_clusters_range[0], n_clusters_range[1]):
            try:
                print(f"🔍 Probando con {k} clusters...")
                
                # Ajustar parámetros de BIRCH para mejor rendimiento
                birch = Birch(
                    n_clusters=k, 
                    threshold=0.1,  # Reducir threshold para más sensibilidad
                    branching_factor=50
                )
                labels = birch.fit_predict(self.X_scaled)
                
                # Verificar que se crearon clusters válidos
                unique_labels = len(set(labels))
                print(f"   Clusters generados: {unique_labels}")
                print(f"   Distribución: {dict(zip(*np.unique(labels, return_counts=True)))}")
                
                if unique_labels < 2:
                    print(f"   ⚠️ Solo se generó {unique_labels} cluster, probando con threshold más bajo...")
                    # Intentar con threshold más bajo
                    birch_low = Birch(n_clusters=k, threshold=0.01, branching_factor=50)
                    labels = birch_low.fit_predict(self.X_scaled)
                    unique_labels = len(set(labels))
                    if unique_labels < 2:
                        continue
                    birch = birch_low
                    
                # Llamar a _calculate_metrics
                result = self._calculate_metrics(labels, self.X_scaled, 'BIRCH',
                                        {'n_clusters': k, 'threshold': birch.threshold})
                
                if result is None:
                    print(f"   ❌ _calculate_metrics retornó None")
                    continue
                    
                print(f"   📊 Silhouette Score: {result['Silhouette_Score']:.4f}")
                
                if result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                    best_labels = labels
                    best_model = birch
                    print(f"   ✅ Nueva mejor configuración encontrada (Score: {best_score:.4f})")
                    
            except Exception as e:
                print(f"   ❌ Error con k={k}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if best_result:
            # Actualizar el resultado con las etiquetas correctas
            best_result['Labels'] = best_labels
            #best_result['Model'] = best_model
            
            self.results_comparison.append(best_result)
            self.df_rfm['BIRCH_Cluster'] = best_labels
            
            print(f"✅ BIRCH Clustering completado - Mejor configuración: {best_result['Parameters']}")
            print(f"📊 Silhouette Score: {best_result['Silhouette_Score']:.4f}")
            print(f"🎯 Número de clusters: {best_result['N_Clusters']}")
            print(f"📈 Distribución final: {dict(zip(*np.unique(best_labels, return_counts=True)))}")
        else:
            print("❌ No se pudo completar BIRCH Clustering")
            print("💡 Sugerencias:")
            print("   - Verificar que los datos estén correctamente escalados")
            print("   - Probar con un rango de clusters diferente")
            print("   - Revisar si hay valores NaN o infinitos en los datos")
        
        return best_result

    def apply_affinity_propagation(self):
        """Aplicar Affinity Propagation (determina automáticamente clusters)"""
        print("🔄 Aplicando Affinity Propagation...")
    
        try:
            best_result = None
            best_score = -1
        
            for damping in [0.5, 0.7, 0.9]:
                try:
                    af = AffinityPropagation(damping=damping, random_state=42, max_iter=300)
                    labels = af.fit_predict(self.X_scaled)
                
                    n_clusters = len(np.unique(labels))
                    if n_clusters > 1 and n_clusters < len(self.X_scaled) * 0.5:
                        result = self._calculate_metrics(labels, self.X_scaled, 'Affinity Propagation',
                                                   {'damping': damping, 'n_clusters': n_clusters})
                        if result and result['Silhouette_Score'] > best_score:
                            best_score = result['Silhouette_Score']
                            best_result = result
                except Exception as e:
                    continue
        
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm['AffinityProp_Cluster'] = best_result['Labels']
                print(f"✅ Affinity Propagation completado")
            return best_result
        
        except Exception as e:
            print(f"❌ Error en Affinity Propagation: {str(e)}")
            return None

    def apply_kmeans_plus_density(self, k_range=(4, 8)):
        """Algoritmo Híbrido: K-Means + Filtro de Densidad"""
        print("🔄 Aplicando K-Means + Filtro de Densidad (Híbrido)...")
    
        best_result = None
        best_score = -1
    
        for k in range(k_range[0], k_range[1]):
            try:
                # Paso 1: K-Means inicial
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                initial_labels = kmeans.fit_predict(self.X_scaled)
            
                # Paso 2: Calcular densidad de cada punto
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=5).fit(self.X_scaled)
                distances, indices = nbrs.kneighbors(self.X_scaled)
                density_scores = 1 / (distances.mean(axis=1) + 1e-10)
            
                # Paso 3: Refinar clusters basado en densidad
                refined_labels = initial_labels.copy()
                density_threshold = np.percentile(density_scores, 25)  # Bottom 25%
            
                # Reasignar puntos de baja densidad al cluster más cercano de alta densidad
                for i, (label, density) in enumerate(zip(initial_labels, density_scores)):
                    if density < density_threshold:
                        # Encontrar el punto de alta densidad más cercano
                        high_density_mask = density_scores >= density_threshold
                        if np.any(high_density_mask):
                            high_density_points = self.X_scaled[high_density_mask]
                            distances_to_high = np.linalg.norm(
                                high_density_points - self.X_scaled[i], axis=1
                            )
                            closest_high_idx = np.where(high_density_mask)[0][np.argmin(distances_to_high)]
                            refined_labels[i] = initial_labels[closest_high_idx]
            
                result = self._calculate_metrics(refined_labels, self.X_scaled, 'K-Means + Density',
                                           {'n_clusters': k, 'density_filter': True})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                
            except Exception as e:
                print(f"⚠️ Error con k={k}: {str(e)}")
                continue
    
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['KMeansDensity_Cluster'] = best_result['Labels']
            print(f"✅ K-Means + Density completado")
        return best_result

    def apply_ensemble_clustering(self, k_range=(4, 6)):
        """Algoritmo Híbrido: Ensemble de múltiples algoritmos"""
        print("🔄 Aplicando Ensemble Clustering (Híbrido)...")
    
        best_result = None
        best_score = -1
    
        for k in range(k_range[0], k_range[1]):
            try:
                # Aplicar múltiples algoritmos
                algorithms = [
                    KMeans(n_clusters=k, random_state=42),
                    AgglomerativeClustering(n_clusters=k),
                    SpectralClustering(n_clusters=k, random_state=42)
                ]
            
                all_labels = []
                for algo in algorithms:
                    try:
                        labels = algo.fit_predict(self.X_scaled)
                        all_labels.append(labels)
                    except:
                        continue
            
                if len(all_labels) < 2:
                    continue
                
                # Crear matriz de co-asociación
                n_samples = len(self.X_scaled)
                co_association = np.zeros((n_samples, n_samples))
            
                for labels in all_labels:
                    for i in range(n_samples):
                        for j in range(n_samples):
                            if labels[i] == labels[j]:
                                co_association[i, j] += 1
            
                # Normalizar por número de algoritmos
                co_association /= len(all_labels)
            
                # Clustering jerárquico sobre la matriz de co-asociación
                distance_matrix = 1 - co_association
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import squareform
            
                linkage_matrix = linkage(squareform(distance_matrix), method='average')
                ensemble_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            
                result = self._calculate_metrics(ensemble_labels, self.X_scaled, 'Ensemble Clustering',
                                           {'n_clusters': k, 'n_algorithms': len(all_labels)})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                
            except Exception as e:
                print(f"⚠️ Error con k={k}: {str(e)}")
                continue
    
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['Ensemble_Cluster'] = best_result['Labels']
            print(f"✅ Ensemble Clustering completado")
        return best_result

    def apply_adaptive_kmeans(self, k_range=(4, 8)):
        """Algoritmo Híbrido: K-Means Adaptativo con refinamiento iterativo"""
        print("🔄 Aplicando K-Means Adaptativo (Híbrido)...")
    
        best_result = None
        best_score = -1
    
        for k in range(k_range[0], k_range[1]):
            try:
                # Paso 1: K-Means inicial
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.X_scaled)
            
                # Paso 2: Refinamiento adaptativo
                improved = True
                iterations = 0
                max_iterations = 5
            
                while improved and iterations < max_iterations:
                    improved = False
                    iterations += 1
                
                    # Calcular varianza intra-cluster para cada cluster
                    cluster_variances = []
                    for cluster_id in range(k):
                        mask = labels == cluster_id
                        if np.sum(mask) > 1:
                            cluster_data = self.X_scaled[mask]
                            variance = np.var(cluster_data, axis=0).mean()
                            cluster_variances.append(variance)
                        else:
                            cluster_variances.append(0)
                
                    # Identificar cluster con mayor varianza
                    if len(cluster_variances) > 0:
                        max_var_cluster = np.argmax(cluster_variances)
                    
                        # Si la varianza es muy alta, intentar re-asignar puntos
                        if cluster_variances[max_var_cluster] > np.mean(cluster_variances) * 1.5:
                            mask = labels == max_var_cluster
                            cluster_points = self.X_scaled[mask]
                        
                            if len(cluster_points) > 2:
                                # Sub-clustering del cluster más variable
                                sub_kmeans = KMeans(n_clusters=2, random_state=42)
                                sub_labels = sub_kmeans.fit_predict(cluster_points)
                            
                                # Reasignar etiquetas
                                point_indices = np.where(mask)[0]
                                for i, sub_label in enumerate(sub_labels):
                                    if sub_label == 1:  # Mover algunos puntos al cluster más cercano
                                        point_idx = point_indices[i]
                                        point = self.X_scaled[point_idx]
                                    
                                        # Encontrar cluster más cercano (excluyendo el actual)
                                        distances = []
                                        for other_cluster in range(k):
                                            if other_cluster != max_var_cluster:
                                                other_mask = labels == other_cluster
                                                if np.sum(other_mask) > 0:
                                                    other_center = np.mean(self.X_scaled[other_mask], axis=0)
                                                    dist = np.linalg.norm(point - other_center)
                                                    distances.append((dist, other_cluster))
                                    
                                        if distances:
                                            closest_cluster = min(distances, key=lambda x: x[0])[1]
                                            labels[point_idx] = closest_cluster
                                            improved = True
            
                result = self._calculate_metrics(labels, self.X_scaled, 'Adaptive K-Means',
                                           {'n_clusters': k, 'iterations': iterations})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
                
            except Exception as e:
                print(f"⚠️ Error con k={k}: {str(e)}")
                continue
    
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['AdaptiveKMeans_Cluster'] = best_result['Labels']
            print(f"✅ Adaptive K-Means completado")
        return best_result

    def apply_hierarchical_subdivision(self, k_range=(6, 20)):
        """Híbrido: Subdivisión Jerárquica Inteligente - Crea más clusters granulares"""
        print("🔄 Aplicando Subdivisión Jerárquica Inteligente...")
    
        best_result = None
        best_score = -1
    
        for target_clusters in range(k_range[0], k_range[1]):
            try:
                # Paso 1: Clustering inicial con pocos clusters
                initial_kmeans = KMeans(n_clusters=3, random_state=42)
                initial_labels = initial_kmeans.fit_predict(self.X_scaled)
            
                # Paso 2: Subdividir cada cluster inicial
                final_labels = np.zeros_like(initial_labels)
                current_label = 0
                subdivision_info = {}
            
                for cluster_id in range(3):
                    mask = initial_labels == cluster_id
                    cluster_data = self.X_scaled[mask]
                    cluster_size = len(cluster_data)
                
                    if cluster_size < 10:  # Cluster muy pequeño, no subdividir
                        final_labels[mask] = current_label
                        subdivision_info[cluster_id] = {'subclusters': 1, 'size': cluster_size}
                        current_label += 1
                    else:
                        # Determinar número de subclusters basado en tamaño y varianza
                        cluster_variance = np.var(cluster_data, axis=0).mean()
                        base_subclusters = max(2, min(6, cluster_size // 20))
                    
                        # Ajustar por varianza (más varianza = más subclusters)
                        variance_factor = min(2.0, cluster_variance / np.var(self.X_scaled, axis=0).mean())
                        n_subclusters = int(base_subclusters * variance_factor)
                        n_subclusters = min(n_subclusters, target_clusters - current_label)
                    
                        if n_subclusters <= 1:
                            final_labels[mask] = current_label
                            current_label += 1
                        else:
                            # Sub-clustering
                            sub_kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                            sub_labels = sub_kmeans.fit_predict(cluster_data)
                        
                            # Asignar etiquetas finales
                            cluster_indices = np.where(mask)[0]
                            for i, sub_label in enumerate(sub_labels):
                                final_labels[cluster_indices[i]] = current_label + sub_label
                        
                            subdivision_info[cluster_id] = {'subclusters': n_subclusters, 'size': cluster_size}
                            current_label += n_subclusters
                
                    if current_label >= target_clusters:
                        break
            
                # Verificar que tenemos el número correcto de clusters
                n_final_clusters = len(np.unique(final_labels))
                if n_final_clusters >= 3:  # Mínimo aceptable
                    result = self._calculate_metrics(final_labels, self.X_scaled, 'Hierarchical Subdivision',
                                               {'target_clusters': target_clusters, 'final_clusters': n_final_clusters,
                                                'subdivision_info': str(subdivision_info)})
                    if result and result['Silhouette_Score'] > best_score:
                        best_score = result['Silhouette_Score']
                        best_result = result
                    
            except Exception as e:
                print(f"⚠️ Error con target_clusters={target_clusters}: {str(e)}")
                continue
    
        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['HierarchicalSub_Cluster'] = best_result['Labels']
            print(f"✅ Subdivisión Jerárquica completada")
        return best_result

    def apply_network_based_segmentation(self):
        """Segmentación basada en redes de similitud entre clientes"""
        print("🔄 Aplicando Segmentación Basada en Redes...")
    
        try:
            from sklearn.neighbors import kneighbors_graph
            from sklearn.cluster import SpectralClustering
            import networkx as nx
        
            # Crear grafo de k-vecinos más cercanos
            n_neighbors = min(10, len(self.X_scaled) // 5)
            knn_graph = kneighbors_graph(self.X_scaled, n_neighbors=n_neighbors, 
                                   mode='connectivity', include_self=False)
        
            # Convertir a NetworkX para análisis de comunidades
            G = nx.from_scipy_sparse_array(knn_graph)
        
            # Detectar comunidades (clusters) usando diferentes algoritmos
            best_result = None
            best_score = -1
        
            algorithms = [
                ('Louvain', nx.community.louvain_communities),
                ('Greedy Modularity', nx.community.greedy_modularity_communities),
            ]
        
            for algo_name, algo_func in algorithms:
                try:
                    if algo_name == 'Louvain':
                        communities = algo_func(G, seed=42)
                    else:
                        communities = algo_func(G)
                
                    # Convertir comunidades a labels
                    labels = np.zeros(len(self.X_scaled))
                    for i, community in enumerate(communities):
                        for node in community:
                            if node < len(labels):
                                labels[node] = i
                
                    n_clusters = len(communities)
                    if n_clusters > 1:
                        result = self._calculate_metrics(labels, self.X_scaled, f'Network-{algo_name}',
                                                   {'n_clusters': n_clusters, 'n_neighbors': n_neighbors})
                        if result and result['Silhouette_Score'] > best_score:
                            best_score = result['Silhouette_Score']
                            best_result = result
                        
                except Exception as e:
                    print(f"⚠️ Error con {algo_name}: {str(e)}")
                    continue
        
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm['Network_Cluster'] = best_result['Labels']
                print(f"✅ Segmentación basada en Redes completada")
            return best_result
        
        except Exception as e:
            print(f"❌ Error en Segmentación basada en Redes: {str(e)}")
            return None

    def pso_optimize_centers(self, X, c, n_particles=20, max_iter=50):
        """Optimizar centros usando Particle Swarm Optimization"""
        n_features = X.shape[1]

        # Inicializar partículas (posiciones de centros)
        particles = np.random.rand(n_particles, c, n_features)
        velocities = np.random.rand(n_particles, c, n_features) * 0.1

        # Normalizar partículas al rango de los datos
        data_min, data_max = X.min(axis=0), X.max(axis=0)
        for i in range(n_particles):
            for j in range(c):
                particles[i, j] = data_min + (data_max - data_min) * particles[i, j]

        personal_best = particles.copy()
        personal_best_scores = np.full(n_particles, float('inf'))
        global_best = None
        global_best_score = float('inf')

        # Parámetros PSO
        w = 0.9  # inercia
        c1, c2 = 2.0, 2.0  # coeficientes de aceleración

        for iteration in range(max_iter):
            for i in range(n_particles):
                # Evaluar fitness (suma de distancias intra-cluster)
                centers = particles[i]
                distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                closest_centers = np.argmin(distances, axis=1)

                fitness = 0
                for cluster_id in range(c):
                    cluster_points = X[closest_centers == cluster_id]
                    if len(cluster_points) > 0:
                        fitness += np.sum(np.linalg.norm(cluster_points - centers[cluster_id], axis=1))

                # Actualizar personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best[i] = particles[i].copy()

                # Actualizar global best
                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best = particles[i].copy()

            # Actualizar velocidades y posiciones
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i]))
                particles[i] += velocities[i]

            # Reducir inercia
            w *= 0.99

        return global_best

    def apply_fuzzy_cmeans_pso(self, c_range=(4, 15), n_particles=20, max_iter=50):
        """Aplicar Fuzzy C-Means con inicialización de centros optimizada por PSO"""
        print("🔄 Aplicando Fuzzy C-Means + PSO...")

        best_result = None
        best_score = -1

        for c in range(c_range[0], c_range[1]):
            try:
                # Optimizar centros iniciales con PSO
                optimized_centers = self.pso_optimize_centers(self.X_scaled, c, n_particles, max_iter)

                # Aplicar Fuzzy C-Means con centros optimizados
                X_T = self.X_scaled.T
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    X_T, c=c, m=2.0, error=0.005, maxiter=1000, init=optimized_centers.T
                )
                labels = np.argmax(u, axis=0)

                result = self._calculate_metrics(labels, self.X_scaled, 'Fuzzy C-Means + PSO',
                                               {'c': c, 'fpc': round(fpc, 4), 'pso_particles': n_particles})
                if result and result['Silhouette_Score'] > best_score:
                    best_score = result['Silhouette_Score']
                    best_result = result
            except Exception as e:
                continue

        if best_result:
            self.results_comparison.append(best_result)
            self.df_rfm['FuzzyCMeans_PSO_Cluster'] = best_result['Labels']
            print(f"✅ Fuzzy C-Means + PSO completado")
        return best_result

    def apply_qlearning_de_kmeans(self, k_range=(4, 11), population_size=20, generations=30, q_episodes=30):
        """
        Aplicar Q-learning based Differential Evolution + K-means
        Optimización de parámetros de clustering usando aprendizaje por refuerzo
        Basado en: Customer segmentation in digital marketing using Q-learning DE + K-means
        """
        print("🔄 Aplicando Q-Learning DE + K-means...")
        
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            import random
            from collections import defaultdict
            
            # Clase Q-Learning Agent para optimización de parámetros
            class QLearningDEAgent:
                def __init__(self, n_actions=4, learning_rate=0.1, epsilon=0.1, discount=0.9):
                    self.q_table = defaultdict(lambda: np.zeros(n_actions))
                    self.learning_rate = learning_rate
                    self.epsilon = epsilon
                    self.discount = discount
                    self.actions = ['mutate_high', 'mutate_low', 'crossover_best', 'crossover_random']
                    
                def get_state(self, fitness_variance, generation_progress, diversity):
                    """Codificar estado del algoritmo DE"""
                    fitness_state = 'high' if fitness_variance > 0.1 else 'low'
                    progress_state = 'early' if generation_progress < 0.3 else 'mid' if generation_progress < 0.7 else 'late'
                    diversity_state = 'diverse' if diversity > 0.5 else 'converged'
                    return f"{fitness_state}_{progress_state}_{diversity_state}"
                
                def choose_action(self, state):
                    """Seleccionar acción usando epsilon-greedy"""
                    if random.random() < self.epsilon:
                        return random.randint(0, len(self.actions) - 1)
                    return np.argmax(self.q_table[state])
                
                def update_q_table(self, state, action, reward, next_state):
                    """Actualizar tabla Q"""
                    current_q = self.q_table[state][action]
                    max_next_q = np.max(self.q_table[next_state])
                    new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
                    self.q_table[state][action] = new_q
            
            # Clase Differential Evolution mejorada con Q-Learning
            class QLearningDE:
                def __init__(self, objective_func, bounds, pop_size=20, agent=None):
                    self.objective_func = objective_func
                    self.bounds = bounds
                    self.pop_size = pop_size
                    self.dimension = len(bounds)
                    self.agent = agent
                    
                    # Parámetros adaptativos
                    self.F_base = 0.5  # Factor de mutación base
                    self.CR_base = 0.9  # Probabilidad de crossover base
                    
                def initialize_population(self):
                    """Inicializar población aleatoria"""
                    population = []
                    for _ in range(self.pop_size):
                        individual = []
                        for low, high in self.bounds:
                            individual.append(random.uniform(low, high))
                        population.append(individual)
                    return population
                
                def mutate(self, population, target_idx, strategy='best'):
                    """Mutación adaptativa basada en Q-learning"""
                    target = population[target_idx]
                    mutant = target.copy()
                    
                    if strategy == 'mutate_high':
                        # Mutación agresiva
                        F = self.F_base * 1.5
                        indices = [i for i in range(len(population)) if i != target_idx]
                        r1, r2, r3 = random.sample(indices, 3)
                        
                        for j in range(self.dimension):
                            mutant[j] = population[r1][j] + F * (population[r2][j] - population[r3][j])
                            mutant[j] = max(self.bounds[j][0], min(self.bounds[j][1], mutant[j]))
                            
                    elif strategy == 'mutate_low':
                        # Mutación conservadora
                        F = self.F_base * 0.5
                        best_idx = 0  # Asumir que está ordenado por fitness
                        indices = [i for i in range(len(population)) if i != target_idx]
                        r1, r2 = random.sample(indices, 2)
                        
                        for j in range(self.dimension):
                            mutant[j] = population[best_idx][j] + F * (population[r1][j] - population[r2][j])
                            mutant[j] = max(self.bounds[j][0], min(self.bounds[j][1], mutant[j]))
                    
                    return mutant
                
                def crossover(self, target, mutant, strategy='crossover_best'):
                    """Crossover adaptativo"""
                    trial = target.copy()
                    
                    if strategy == 'crossover_best':
                        CR = self.CR_base
                    elif strategy == 'crossover_random':
                        CR = random.uniform(0.1, 1.0)
                    else:
                        CR = self.CR_base
                    
                    for j in range(self.dimension):
                        if random.random() < CR or j == random.randint(0, self.dimension - 1):
                            trial[j] = mutant[j]
                    
                    return trial
                
                def calculate_diversity(self, population):
                    """Calcular diversidad de la población"""
                    if len(population) < 2:
                        return 0.0
                    
                    distances = []
                    for i in range(len(population)):
                        for j in range(i + 1, len(population)):
                            dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
                            distances.append(dist)
                    
                    return np.mean(distances) if distances else 0.0
                
                def optimize(self, max_generations):
                    """Optimización principal con Q-learning"""
                    population = self.initialize_population()
                    fitness_history = []
                    
                    for generation in range(max_generations):
                        # Evaluar población
                        fitness_scores = [self.objective_func(ind) for ind in population]
                        
                        # Ordenar por fitness (mayor es mejor para silhouette)
                        sorted_indices = np.argsort(fitness_scores)[::-1]
                        population = [population[i] for i in sorted_indices]
                        fitness_scores = [fitness_scores[i] for i in sorted_indices]
                        
                        fitness_history.append(fitness_scores[0])
                        
                        # Calcular métricas para Q-learning
                        fitness_variance = np.var(fitness_scores)
                        generation_progress = generation / max_generations
                        diversity = self.calculate_diversity(population)
                        
                        # Estado para Q-learning
                        current_state = self.agent.get_state(fitness_variance, generation_progress, diversity)
                        
                        # Nueva población
                        new_population = [population[0]]  # Mantener el mejor (elitismo)
                        
                        for i in range(1, self.pop_size):
                            # Q-learning decide la estrategia
                            action = self.agent.choose_action(current_state)
                            strategy = self.agent.actions[action]
                            
                            # Aplicar mutación y crossover
                            if 'mutate' in strategy:
                                mutant = self.mutate(population, i, strategy)
                                trial = self.crossover(population[i], mutant, 'crossover_best')
                            else:
                                mutant = self.mutate(population, i, 'mutate_low')
                                trial = self.crossover(population[i], mutant, strategy)
                            
                            # Evaluar trial
                            trial_fitness = self.objective_func(trial)
                            
                            # Selección
                            if trial_fitness > fitness_scores[i]:
                                new_population.append(trial)
                                reward = 1.0  # Recompensa positiva
                            else:
                                new_population.append(population[i])
                                reward = -0.1  # Penalización pequeña
                            
                            # Actualizar Q-table
                            next_state = self.agent.get_state(fitness_variance, generation_progress, diversity)
                            self.agent.update_q_table(current_state, action, reward, next_state)
                        
                        population = new_population
                        
                        # Log progreso
                        if generation % 10 == 0:
                            print(f"      Generación {generation}: Mejor fitness = {fitness_scores[0]:.4f}")
                    
                    return population[0], fitness_scores[0]
            
            best_result = None
            best_score = -1
            
            print("   🧠 Inicializando Q-Learning Agent...")
            
            for k in range(k_range[0], k_range[1]):
                try:
                    print(f"   🎯 Optimizando para k={k}...")
                    
                    # Función objetivo para DE
                    def objective_function(params):
                        """Función objetivo que optimiza parámetros de K-means"""
                        try:
                            # Decodificar parámetros
                            n_init, max_iter, tol_exp = params
                            n_init = max(1, int(n_init))
                            max_iter = max(10, int(max_iter))
                            tol = 10 ** (-abs(tol_exp))  # Tolerancia logarítmica
                            
                            # K-means con parámetros optimizados
                            kmeans = KMeans(
                                n_clusters=k,
                                n_init=n_init,
                                max_iter=max_iter,
                                tol=tol,
                                random_state=42
                            )
                            
                            labels = kmeans.fit_predict(self.X_scaled)
                            
                            # Calcular fitness (Silhouette Score)
                            if len(set(labels)) > 1:
                                return silhouette_score(self.X_scaled, labels)
                            else:
                                return -1.0
                                
                        except Exception as e:
                            return -1.0  # Penalización por error
                    
                    # Bounds para parámetros: [n_init, max_iter, tol_exponent]
                    bounds = [
                        (3, 20),      # n_init: número de inicializaciones
                        (100, 500),   # max_iter: iteraciones máximas
                        (2, 8)        # tol_exponent: exponente para tolerancia
                    ]
                    
                    # Crear agente Q-learning
                    q_agent = QLearningDEAgent()
                    
                    # Optimizar con DE + Q-learning
                    de_optimizer = QLearningDE(
                        objective_func=objective_function,
                        bounds=bounds,
                        pop_size=population_size,
                        agent=q_agent
                    )
                    
                    best_params, best_fitness = de_optimizer.optimize(generations)
                    
                    # Aplicar mejores parámetros encontrados
                    n_init_opt = max(1, int(best_params[0]))
                    max_iter_opt = max(10, int(best_params[1]))
                    tol_opt = 10 ** (-abs(best_params[2]))
                    
                    # K-means final con parámetros optimizados
                    kmeans_final = KMeans(
                        n_clusters=k,
                        n_init=n_init_opt,
                        max_iter=max_iter_opt,
                        tol=tol_opt,
                        random_state=42
                    )
                    
                    labels_final = kmeans_final.fit_predict(self.X_scaled)
                    
                    # Calcular métricas
                    result = self._calculate_metrics(
                        labels_final,
                        self.X_scaled,
                        'Q-Learning-DE-KMeans',
                        {
                            'k': k,
                            'n_init_opt': n_init_opt,
                            'max_iter_opt': max_iter_opt,
                            'tol_opt': f"{tol_opt:.2e}",
                            'de_generations': generations,
                            'q_episodes': q_episodes
                        }
                    )
                    
                    if result and result['Silhouette_Score'] > best_score:
                        best_score = result['Silhouette_Score']
                        best_result = result
                    
                    print(f"   📈 k={k}: Score={result['Silhouette_Score']:.4f} (Parámetros optimizados)" if result else f"   ❌ k={k}: Error en métricas")
                    
                except Exception as e:
                    print(f"   ⚠️  Error con k={k}: {str(e)}")
                    continue
            
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm['QLearningDE_Cluster'] = best_result['Labels']
                print(f"✅ Q-Learning DE + K-means completado - Mejor configuración: {best_result['Parameters']}")
                print(f"   🎯 Mejor Silhouette Score: {best_result['Silhouette_Score']:.4f}")
                print(f"   📊 Clusters encontrados: {best_result['N_Clusters']}")
                print(f"   🤖 Parámetros optimizados automáticamente por Q-Learning")
            else:
                print("❌ No se pudo completar el Q-Learning DE + K-means")
            
            return best_result
            
        except Exception as e:
            print(f"❌ Error en Q-Learning DE + K-means: {str(e)}")
            return None

    def apply_reinforcement_clustering_alternative(self, k_range=(4, 11), episodes=100):
        """
        Alternativa simplificada de clustering con aprendizaje por refuerzo
        Optimiza directamente los centroides usando Q-learning
        """
        print("🔄 Aplicando Reinforcement Learning Clustering (Alternativo)...")
        
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            import random
            from collections import defaultdict
            
            class CentroidQLearning:
                """Q-Learning para optimización directa de centroides"""
                
                def __init__(self, n_features, n_clusters, learning_rate=0.1):
                    self.n_features = n_features
                    self.n_clusters = n_clusters
                    self.learning_rate = learning_rate
                    self.centroids = np.random.randn(n_clusters, n_features)
                    
                def assign_clusters(self, X):
                    """Asignar puntos a clusters más cercanos"""
                    distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
                    return np.argmin(distances, axis=1)
                
                def update_centroids(self, X, labels):
                    """Actualizar centroides basado en asignaciones"""
                    new_centroids = np.zeros_like(self.centroids)
                    for k in range(self.n_clusters):
                        mask = labels == k
                        if mask.sum() > 0:
                            new_centroids[k] = X[mask].mean(axis=0)
                        else:
                            new_centroids[k] = self.centroids[k]  # Mantener si no hay puntos
                    
                    # Actualización con learning rate
                    self.centroids = (1 - self.learning_rate) * self.centroids + self.learning_rate * new_centroids
                
                def get_reward(self, X, labels):
                    """Calcular recompensa basada en cohesión intra-cluster"""
                    try:
                        if len(set(labels)) > 1:
                            return silhouette_score(X, labels)
                        else:
                            return -1.0
                    except:
                        return -1.0
            
            best_result = None
            best_score = -1
            
            for k in range(k_range[0], k_range[1]):
                try:
                    print(f"   🎯 RL Clustering para k={k}...")
                    
                    # Inicializar agente Q-learning
                    rl_agent = CentroidQLearning(
                        n_features=self.X_scaled.shape[1],
                        n_clusters=k,
                        learning_rate=0.1
                    )
                    
                    best_episode_score = -1
                    best_labels = None
                    
                    # Entrenamiento por episodios
                    for episode in range(episodes):
                        # Asignar clusters
                        labels = rl_agent.assign_clusters(self.X_scaled)
                        
                        # Calcular recompensa
                        reward = rl_agent.get_reward(self.X_scaled, labels)
                        
                        # Actualizar centroides
                        rl_agent.update_centroids(self.X_scaled, labels)
                        
                        # Guardar mejor resultado
                        if reward > best_episode_score:
                            best_episode_score = reward
                            best_labels = labels.copy()
                        
                        # Log cada 20 episodios
                        if episode % 20 == 0:
                            print(f"      Episodio {episode}: Reward={reward:.4f}")
                    
                    # Usar las mejores etiquetas encontradas
                    if best_labels is not None:
                        result = self._calculate_metrics(
                            best_labels,
                            self.X_scaled,
                            'RL-Clustering',
                            {
                                'k': k,
                                'episodes': episodes,
                                'final_reward': best_episode_score
                            }
                        )
                        
                        if result and result['Silhouette_Score'] > best_score:
                            best_score = result['Silhouette_Score']
                            best_result = result
                        
                        print(f"   📈 k={k}: Score={result['Silhouette_Score']:.4f}" if result else f"   ❌ k={k}: Error en métricas")
                    
                except Exception as e:
                    print(f"   ⚠️  Error con k={k}: {str(e)}")
                    continue
            
            if best_result:
                self.results_comparison.append(best_result)
                self.df_rfm['RL_Cluster'] = best_result['Labels']
                print(f"✅ RL Clustering completado - Mejor k: {best_result['Parameters']}")
            
            return best_result
            
        except Exception as e:
            print(f"❌ Error en RL Clustering: {str(e)}")
            return None

    def run_all_algorithms(self):
        """Ejecutar todos los algoritmos de clustering"""
        print("🚀 Iniciando análisis completo de clustering...")
        
        if self.df_rfm is None:
            self.prepare_rfm_data()
        
        # Ejecutar todos los algoritmos
  
        #self.apply_qlearning_de_kmeans()
        #self.apply_reinforcement_clustering_alternative
        self.apply_kmeans()
        self.apply_fuzzy_cmeans()
        self.apply_dbscan()
        #self.apply_optics()
        self.apply_hdbscan()
        self.apply_meanshift()
        #self.apply_gaussian_mixture()
        self.apply_spectral_clustering()
        self.apply_birch_clustering()
        #self.apply_affinity_propagation() No sirve 
        self.apply_ensemble_clustering()
        #self.apply_kmeans_plus_density()
        #self.apply_adaptive_kmeans()
        #self.apply_hierarchical_subdivision()
        #self.apply_network_based_segmentation()
        #self.apply_fixed_fuzzy_cmeans_pso(c_range=(5, 12), n_particles=30, max_iter=100)
        #self.apply_enhanced_deep_embedded_clustering()
        self.apply_optimized_variational_deep_embedding()
        #self.apply_hybrid_firefly_clustering()  mucho costo computacional
        #self.apply_graph_clustering_alternative()
        self.apply_gnn_clustering()
        self.apply_gnn_guided_hdbscan()

        

    
        print(f"✅ Análisis completado! {len(self.results_comparison)} algoritmos ejecutados")
        return self.get_results_comparison()
    
    #============================= ANALIZAR RESULTADOS algoritmos ==========================================#
    def get_results_comparison(self):
        """Obtener tabla comparativa de resultados"""
        if not self.results_comparison:
            print("⚠️ No hay resultados para comparar. Ejecuta run_all_algorithms() primero.")
            return None
            
        df_comparison = pd.DataFrame(self.results_comparison)
        
        # Eliminar columna de labels para visualización
        df_display = df_comparison.drop('Labels', axis=1, errors='ignore')
        
        # Ordenar por Silhouette Score
        df_display = df_display.sort_values('Silhouette_Score', ascending=False)
        
        return df_display
    
    
        
        # Calcular grid size
    def _get_pca_coordinates(self):
        """Obtener coordenadas PCA para visualización 2D"""
        pca = PCA(n_components=2)
        # Usar los datos normalizados en lugar de self.data[['R', 'F', 'M']]
        coords = pca.fit_transform(self.X_scaled)
        print(f"📊 PCA - Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")
        return coords
    def _get_tsne_coordinates(self):
        """Obtener coordenadas t-SNE para visualización 2D"""
        print("🔄 Calculando t-SNE (puede tomar unos segundos)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        # Usar los datos normalizados en lugar de self.data[['R', 'F', 'M']]
        coords = tsne.fit_transform(self.X_scaled)
        return coords
    def plot_all_algorithms_grid(self, method='t-sne', figsize=(20, 25)):
            """
            Crear grid con scatter plots de todos los algoritmos
            
            Parameters:
            method: 'pca' o 'tsne' para reducción de dimensionalidad
            figsize: Tamaño de la figura
            """
            n_algorithms = len(self.results_comparison)
            
            # Calcular grid size
            cols = 3
            rows = (n_algorithms + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            axes = axes.flatten() if n_algorithms > 1 else [axes]
            
            # Preparar datos según método
            if method == 'pca':
                coords = self._get_pca_coordinates()
                coord_labels = ['PC1', 'PC2']
            else:  # tsne
                coords = self._get_tsne_coordinates()
                coord_labels = ['t-SNE 1', 't-SNE 2']
            
            # Plot cada algoritmo
            for i, result in enumerate(self.results_comparison):
                if i < len(axes):
                    self._plot_single_algorithm(
                        axes[i], 
                        coords, 
                        result, 
                        coord_labels,
                        method
                    )
            
            # Ocultar ejes vacíos
            for i in range(n_algorithms, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()

    def _plot_single_algorithm(self, ax, coords, result, coord_labels, method):
            
            """Plot individual algorithm results"""
            algorithm = result['Algorithm']
            labels = result['Labels']
            unique_clusters = np.unique(labels)
            n_clusters = len(unique_clusters)
            
            # Asignar colores
            colors = plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 12)))
            
            # Plot cada cluster
            for i, cluster in enumerate(unique_clusters):
                mask = labels == cluster
                cluster_coords = coords[mask]
                
                ax.scatter(
                    cluster_coords[:, 0], 
                    cluster_coords[:, 1],
                    c=[colors[i]], 
                    label=f'C{cluster} ({np.sum(mask)})',
                    alpha=0.7,
                    s=30,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            ax.set_xlabel(coord_labels[0])
            ax.set_ylabel(coord_labels[1])
            ax.set_title(f'{algorithm}\n({n_clusters} clusters)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Agregar métricas si están disponibles
            if 'Silhouette_Score' in result:
                silhouette = result['Silhouette_Score']
                if silhouette != 'N/A':
                    ax.text(0.02, 0.98, f'Silhouette: {silhouette:.3f}', 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_summary_report(self):
        """
        Crear reporte resumen de todos los algoritmos
        """
        summary_data = []
        
        for result in self.results_comparison:
            algorithm = result['Algorithm']
            labels = result['Labels']
            unique_clusters = np.unique(labels)
            n_clusters = len(unique_clusters)
            
            # Calcular tamaños de clusters
            cluster_sizes = []
            for cluster in unique_clusters:
                mask = labels == cluster
                cluster_size = np.sum(mask)
                cluster_sizes.append(cluster_size)
            
            # Crear DataFrame temporal para cálculos
            df_temp = self.df_rfm.copy()
            df_temp['Cluster'] = labels
            
            # Calcular varianza intra-cluster
            intra_cluster_variance = []
            for cluster in unique_clusters:
                mask = labels == cluster
                cluster_data = df_temp[mask][['Recencia', 'Frequencia', 'Valor_Monetario']]
                
                if len(cluster_data) > 1:
                    variance = np.var(cluster_data.values)
                    intra_cluster_variance.append(variance)
            
            avg_intra_variance = np.mean(intra_cluster_variance) if intra_cluster_variance else 0
            
            # Recopilar datos del resumen
            summary_data.append({
                'Algorithm': algorithm,
                'N_Clusters': n_clusters,
                'Min_Cluster_Size': min(cluster_sizes),
                'Max_Cluster_Size': max(cluster_sizes),
                'Avg_Cluster_Size': np.mean(cluster_sizes),
                'Std_Cluster_Size': np.std(cluster_sizes),
                'Avg_Intra_Variance': avg_intra_variance,
                'Silhouette_Score': result.get('Silhouette_Score', 'N/A'),
                'Calinski_Harabasz': result.get('Calinski_Harabasz_Score', 'N/A')
            })
        
        summary_df = pd.DataFrame(summary_data)
        return print(summary_df)

    def save_results_comparison(self, filename=None):
        """Guardar comparación de resultados"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clustering_comparison_{timestamp}.csv"
            
        df_comparison = self.get_results_comparison()
        if df_comparison is not None:
            df_comparison.to_csv(filename, index=False)
            print(f"💾 Resultados guardados en: {filename}")
            return filename
        return None
    
    def plot_comparison_metrics(self, figsize=(15, 10)):
        """Visualizar comparación de métricas"""
        if not self.results_comparison:
            print("⚠️ No hay resultados para visualizar.")
            return
            
        df_comp = self.get_results_comparison()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Silhouette Score
        sns.barplot(data=df_comp, x='Algorithm', y='Silhouette_Score', ax=axes[0,0])
        axes[0,0].set_title('Silhouette Score (Mayor es mejor)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin Index
        sns.barplot(data=df_comp, x='Algorithm', y='Davies_Bouldin', ax=axes[0,1])
        axes[0,1].set_title('Davies-Bouldin Index (Menor es mejor)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz Index
        sns.barplot(data=df_comp, x='Algorithm', y='Calinski_Harabasz', ax=axes[1,0])
        axes[1,0].set_title('Calinski-Harabasz Index (Mayor es mejor)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Número de clusters
        sns.barplot(data=df_comp, x='Algorithm', y='N_Clusters', ax=axes[1,1])
        axes[1,1].set_title('Número de Clusters')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_best_algorithm(self):
        """Obtener el mejor algoritmo basado en múltiples métricas"""
        if not self.results_comparison:
            return None
            
        df_comp = pd.DataFrame(self.results_comparison)
        
        # Normalizar métricas (invertir Davies-Bouldin porque menor es mejor)
        scaler = MinMaxScaler()
        metrics_normalized = scaler.fit_transform(pd.DataFrame({
            'silhouette': df_comp['Silhouette_Score'],
            'davies_bouldin_inv': -df_comp['Davies_Bouldin'],  # Invertir
            'calinski_harabasz': df_comp['Calinski_Harabasz']
        }))
        
        # Score compuesto (pesos ajustables)
        composite_scores = (0.4 * metrics_normalized[:, 0] + 
                          0.3 * metrics_normalized[:, 1] + 
                          0.3 * metrics_normalized[:, 2])
        
        best_idx = np.argmax(composite_scores)
        best_algorithm = df_comp.iloc[best_idx]
        
        print(f"🏆 Mejor algoritmo: {best_algorithm['Algorithm']}")
        print(f"   Parámetros: {best_algorithm['Parameters']}")
        print(f"   Silhouette Score: {best_algorithm['Silhouette_Score']}")
        print(f"   Score Compuesto: {composite_scores[best_idx]:.4f}")
        
        return best_algorithm
#======================================= ANALYSIS DE CLUSTERING RFM ========================================#

    def analyze_best_algorithm(self):
        """Identificar y analizar el mejor algoritmo basado en métricas"""
        if not self.results_comparison:
            print("⚠️ No hay resultados para analizar")
            return None
            
        # Crear DataFrame de comparación
        df_comp = pd.DataFrame(self.results_comparison)
        
        # Calcular score compuesto (normalizado)
        df_comp['Silhouette_Norm'] = (df_comp['Silhouette_Score'] - df_comp['Silhouette_Score'].min()) / (df_comp['Silhouette_Score'].max() - df_comp['Silhouette_Score'].min())
        df_comp['DBI_Norm'] = 1 - ((df_comp['Davies_Bouldin'] - df_comp['Davies_Bouldin'].min()) / (df_comp['Davies_Bouldin'].max() - df_comp['Davies_Bouldin'].min()))
        df_comp['CH_Norm'] = (df_comp['Calinski_Harabasz'] - df_comp['Calinski_Harabasz'].min()) / (df_comp['Calinski_Harabasz'].max() - df_comp['Calinski_Harabasz'].min())
        
        # Score compuesto (mayor es mejor)
        df_comp['Composite_Score'] = (df_comp['Silhouette_Norm'] * 0.4 + 
                                     df_comp['DBI_Norm'] * 0.3 + 
                                     df_comp['CH_Norm'] * 0.3)
        
        best_algo = df_comp.loc[df_comp['Composite_Score'].idxmax()]
        
        print("🏆 MEJOR ALGORITMO IDENTIFICADO:")
        print(f"Algoritmo: {best_algo['Algorithm']}")
        print(f"Parámetros: {best_algo['Parameters']}")
        print(f"Número de Clusters: {best_algo['N_Clusters']}")
        print(f"Score Compuesto: {best_algo['Composite_Score']:.4f}")
        print(f"Silhouette Score: {best_algo['Silhouette_Score']:.4f}")
        print(f"Davies-Bouldin: {best_algo['Davies_Bouldin']:.4f}")
        print(f"Calinski-Harabasz: {best_algo['Calinski_Harabasz']:.4f}")
        
        return best_algo
    
    def get_cluster_statistics(self, cluster_column='GNN_Cluster'):
        """Obtener estadísticas descriptivas por cluster"""
        if cluster_column not in self.df_rfm.columns:
            print(f"⚠️ Columna {cluster_column} no encontrada")
            return None
            
        stats_dict = {}
        
        # Estadísticas por cluster
        for cluster in sorted(self.df_rfm[cluster_column].unique()):
            if cluster == -1:  # Ruido en DBSCAN/HDBSCAN
                continue
                
            cluster_data = self.df_rfm[self.df_rfm[cluster_column] == cluster]
            
            stats_dict[f'Cluster_{cluster}'] = {
                'Tamaño': len(cluster_data),
                'Porcentaje': f"{len(cluster_data)/len(self.df_rfm)*100:.2f}%",
                
                # Recencia
                'Recencia_Media': cluster_data['Recencia'].mean(),
                'Recencia_Mediana': cluster_data['Recencia'].median(),
                'Recencia_Std': cluster_data['Recencia'].std(),
                'Recencia_Min': cluster_data['Recencia'].min(),
                'Recencia_Max': cluster_data['Recencia'].max(),
                
                # Frecuencia
                'Frequencia_Media': cluster_data['Frequencia'].mean(),
                'Frequencia_Mediana': cluster_data['Frequencia'].median(),
                'Frequencia_Std': cluster_data['Frequencia'].std(),
                'Frequencia_Min': cluster_data['Frequencia'].min(),
                'Frequencia_Max': cluster_data['Frequencia'].max(),
                
                # Valor Monetario
                'ValorMon_Media': cluster_data['Valor_Monetario'].mean(),
                'ValorMon_Mediana': cluster_data['Valor_Monetario'].median(),
                'ValorMon_Std': cluster_data['Valor_Monetario'].std(),
                'ValorMon_Min': cluster_data['Valor_Monetario'].min(),
                'ValorMon_Max': cluster_data['Valor_Monetario'].max(),
            }
        
        # Convertir a DataFrame para mejor visualización
        stats_df = pd.DataFrame(stats_dict).T
        
        print("📊 ESTADÍSTICAS POR CLUSTER:")
        print("="*60)
        for cluster_name, stats in stats_dict.items():
            print(f"\n{cluster_name.upper()}:")
            print(f"  Tamaño: {stats['Tamaño']} clientes ({stats['Porcentaje']})")
            print(f"  Recencia (días): μ={stats['Recencia_Media']:.1f}, σ={stats['Recencia_Std']:.1f}")
            print(f"  Frecuencia: μ={stats['Frequencia_Media']:.1f}, σ={stats['Frequencia_Std']:.1f}")
            print(f"  Valor Monetario: μ=${stats['ValorMon_Media']:,.0f}, σ=${stats['ValorMon_Std']:,.0f}")
        
        return stats_df
    
    def analyze_cluster_separation(self, cluster_column='GNN_Cluster'):
        """Analizar la separación entre clusters"""
        if cluster_column not in self.df_rfm.columns:
            return None
            
        # Test ANOVA para cada variable RFM
        clusters = self.df_rfm[cluster_column].unique()
        clean_clusters = [c for c in clusters if c != -1]
        
        if len(clean_clusters) < 2:
            print("⚠️ Se necesitan al menos 2 clusters para análisis de separación")
            return None
            
        separation_results = {}
        
        for var in ['Recencia', 'Frequencia', 'Valor_Monetario']:
            groups = []
            for cluster in clean_clusters:
                cluster_data = self.df_rfm[self.df_rfm[cluster_column] == cluster][var]
                groups.append(cluster_data)
            
            # ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Eta-squared (tamaño del efecto)
            ss_between = sum([len(group) * (group.mean() - self.df_rfm[var].mean())**2 for group in groups])
            ss_total = sum([(x - self.df_rfm[var].mean())**2 for x in self.df_rfm[var]])
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            separation_results[var] = {
                'F_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significance': 'Significativo' if p_value < 0.05 else 'No significativo'
            }
        
        print("🔍 ANÁLISIS DE SEPARACIÓN ENTRE CLUSTERS:")
        print("="*50)
        for var, results in separation_results.items():
            print(f"\n{var}:")
            print(f"  F-statistic: {results['F_statistic']:.4f}")
            print(f"  p-value: {results['p_value']:.6f}")
            print(f"  Eta-squared: {results['eta_squared']:.4f}")
            print(f"  Resultado: {results['significance']}")
        
        return separation_results
    
    def generate_business_insights(self, cluster_column='GNN_Cluster'):
        """Generar insights de negocio basados en los clusters"""
        if cluster_column not in self.df_rfm.columns:
            return None
            
        insights = {}
        
        for cluster in sorted(self.df_rfm[cluster_column].unique()):
            if cluster == -1:
                continue
                
            cluster_data = self.df_rfm[self.df_rfm[cluster_column] == cluster]
            
            # Calcular percentiles RFM
            r_percentile = (self.df_rfm['Recencia'] <= cluster_data['Recencia'].mean()).mean()
            f_percentile = (self.df_rfm['Frequencia'] <= cluster_data['Frequencia'].mean()).mean()
            m_percentile = (self.df_rfm['Valor_Monetario'] <= cluster_data['Valor_Monetario'].mean()).mean()
            
            # Clasificar comportamiento
            if r_percentile <= 0.3 and f_percentile >= 0.7 and m_percentile >= 0.7:
                segment_type = "🌟 Champions"
                business_action = "Programas VIP, productos premium, early access"
            elif r_percentile <= 0.5 and f_percentile >= 0.5 and m_percentile >= 0.5:
                segment_type = "💎 Loyal Customers"
                business_action = "Programas de lealtad, cross-selling, upselling"
            elif r_percentile <= 0.5 and f_percentile >= 0.3 and m_percentile <= 0.7:
                segment_type = "⭐ Potential Loyalists"
                business_action = "Incentivos de frecuencia, programas de membresía"
            elif r_percentile >= 0.7 and f_percentile >= 0.5 and m_percentile >= 0.5:
                segment_type = "💰 Big Spenders"
                business_action = "Campañas de reactivación, ofertas especiales"
            elif r_percentile <= 0.3 and f_percentile <= 0.3:
                segment_type = "🆕 New Customers"
                business_action = "Onboarding, educación de producto, incentivos iniciales"
            elif r_percentile >= 0.7 and f_percentile <= 0.3:
                segment_type = "😴 At Risk"
                business_action = "Campañas de retención urgente, descuentos, feedback"
            elif r_percentile >= 0.7 and f_percentile >= 0.3 and m_percentile <= 0.3:
                segment_type = "🔄 Cannot Lose Them"
                business_action = "Win-back campaigns, encuestas, soporte personal"
            else:
                segment_type = "📊 Regular Customers"
                business_action = "Campañas estándar, newsletters, promociones regulares"
            
            insights[f'Cluster_{cluster}'] = {
                'Segment_Type': segment_type,
                'Business_Action': business_action,
                'Customer_Count': len(cluster_data),
                'Avg_Recency': cluster_data['Recencia'].mean(),
                'Avg_Frequency': cluster_data['Frequencia'].mean(),
                'Avg_Monetary': cluster_data['Valor_Monetario'].mean(),
                'Total_Revenue_Potential': cluster_data['Valor_Monetario'].sum()
            }
        
        print("💼 INSIGHTS DE NEGOCIO POR CLUSTER:")
        print("="*60)
        for cluster_name, insight in insights.items():
            print(f"\n{cluster_name} - {insight['Segment_Type']}")
            print(f"  Clientes: {insight['Customer_Count']:,}")
            print(f"  Potencial de ingresos: ${insight['Total_Revenue_Potential']:,.0f}")
            print(f"  Acción recomendada: {insight['Business_Action']}")
        
        return insights
    
    def plot_cluster_analysis(self, cluster_column='GNN_Cluster'):
        """Crear visualizaciones completas del análisis de clusters"""
        if cluster_column not in self.df_rfm.columns:
            return None
            
        # Configurar el estilo
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análisis Completo de Clusters RFM', fontsize=16, fontweight='bold')
        
        # 1. Distribución de clusters
        cluster_counts = self.df_rfm[cluster_column].value_counts().sort_index()
        axes[0,0].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                     autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Distribución de Clusters')
        
        # 2. Box plot Recencia
        data_for_box = [group['Recencia'].values for name, group in self.df_rfm.groupby(cluster_column) if name != -1]
        labels_for_box = [f'C{name}' for name, group in self.df_rfm.groupby(cluster_column) if name != -1]
        axes[0,1].boxplot(data_for_box, labels=labels_for_box)
        axes[0,1].set_title('Recencia por Cluster')
        axes[0,1].set_ylabel('Días')
        
        # 3. Box plot Frecuencia
        data_for_box = [group['Frequencia'].values for name, group in self.df_rfm.groupby(cluster_column) if name != -1]
        axes[0,2].boxplot(data_for_box, labels=labels_for_box)
        axes[0,2].set_title('Frecuencia por Cluster')
        axes[0,2].set_ylabel('Transacciones')
        
        # 4. Box plot Valor Monetario
        data_for_box = [group['Valor_Monetario'].values for name, group in self.df_rfm.groupby(cluster_column) if name != -1]
        axes[1,0].boxplot(data_for_box, labels=labels_for_box)
        axes[1,0].set_title('Valor Monetario por Cluster')
        axes[1,0].set_ylabel('Valor ($)')
        
        # 5. Scatter plot RFM
        axes[1,1].scatter(self.df_rfm['Frequencia'], self.df_rfm['Valor_Monetario'], 
                                   c=self.df_rfm[cluster_column], cmap='tab10', alpha=0.6)
        axes[1,1].set_xlabel('Frecuencia')
        axes[1,1].set_ylabel('Valor Monetario')
        axes[1,1].set_title('Clusters en espacio F-M')
        
        # 6. Heatmap de centroides
        centroids_data = []
        cluster_labels = []
        for cluster in sorted(self.df_rfm[cluster_column].unique()):
            if cluster != -1:
                cluster_data = self.df_rfm[self.df_rfm[cluster_column] == cluster]
                centroids_data.append([
                    cluster_data['Recencia'].mean(),
                    cluster_data['Frequencia'].mean(),
                    cluster_data['Valor_Monetario'].mean()
                ])
                cluster_labels.append(f'Cluster {cluster}')
        
        if centroids_data:
            centroids_df = pd.DataFrame(centroids_data, 
                                      columns=['Recencia', 'Frequencia', 'Valor_Monetario'],
                                      index=cluster_labels)
            # Normalizar para heatmap
            centroids_norm = (centroids_df - centroids_df.min()) / (centroids_df.max() - centroids_df.min())
            
            axes[1,2].imshow(centroids_norm.values, cmap='RdYlBu_r', aspect='auto')
            axes[1,2].set_xticks(range(len(centroids_norm.columns)))
            axes[1,2].set_xticklabels(centroids_norm.columns, rotation=45)
            axes[1,2].set_yticks(range(len(centroids_norm.index)))
            axes[1,2].set_yticklabels(centroids_norm.index)
            axes[1,2].set_title('Heatmap de Centroides (Normalizado)')
            
            # Añadir valores en el heatmap
            for i in range(len(centroids_norm.index)):
                for j in range(len(centroids_norm.columns)):
                    axes[1,2].text(j, i, f'{centroids_norm.iloc[i,j]:.2f}', 
                                  ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.show()
        
    def export_cluster_report(self, cluster_column='KMeans_Cluster', filename='cluster_analysis_report.xlsx'):
        """Exportar reporte completo de análisis de clusters a Excel"""
        if cluster_column not in self.df_rfm.columns:
            return None
            
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Hoja 1: Datos originales con clusters
            self.df_rfm.to_excel(writer, sheet_name='Datos_con_Clusters', index=False)
            
            # Hoja 2: Estadísticas por cluster
            stats_df = self.get_cluster_statistics(cluster_column)
            if stats_df is not None:
                stats_df.to_excel(writer, sheet_name='Estadisticas_Clusters')
            
            # Hoja 3: Comparación de algoritmos
            if self.results_comparison:
                pd.DataFrame(self.results_comparison).drop('Labels', axis=1, errors='ignore').to_excel(
                    writer, sheet_name='Comparacion_Algoritmos', index=False)
            
            # Hoja 4: Insights de negocio
            insights = self.generate_business_insights(cluster_column)
            if insights:
                insights_df = pd.DataFrame(insights).T
                insights_df.to_excel(writer, sheet_name='Insights_Negocio')
        
        print(f"📄 Reporte exportado a: {filename}")
        
    def validate_cluster_quality(self, cluster_column='KMeans_Cluster'):
        """Validación integral de la calidad de clusters"""
        print("🔍 VALIDACIÓN DE CALIDAD DE CLUSTERS")
        print("="*50)
        
        # 1. Tamaño de clusters
        cluster_sizes = self.df_rfm[cluster_column].value_counts().sort_index()
        min_size = cluster_sizes.min()
        max_size = cluster_sizes.max()
        size_ratio = max_size / min_size if min_size > 0 else float('inf')
        
        print(f"📏 Tamaños de clusters:")
        print(f"  Menor: {min_size} | Mayor: {max_size} | Ratio: {size_ratio:.2f}")
        
        size_quality = "✅ Balanceado" if size_ratio <= 5 else "⚠️ Desbalanceado" if size_ratio <= 10 else "❌ Muy desbalanceado"
        print(f"  Evaluación: {size_quality}")
        
        # 2. Separación estadística
        separation = self.analyze_cluster_separation(cluster_column)
        significant_vars = sum([1 for var, result in separation.items() if result['p_value'] < 0.05])
        
        print(f"\n📊 Separación estadística:")
        print(f"  Variables significativas: {significant_vars}/3")
        sep_quality = "✅ Excelente" if significant_vars == 3 else "⚠️ Aceptable" if significant_vars >= 2 else "❌ Pobre"
        print(f"  Evaluación: {sep_quality}")
        
        # 3. Coherencia de negocio
        insights = self.generate_business_insights(cluster_column)
        unique_segments = len(set([insight['Segment_Type'] for insight in insights.values()]))
        
        print(f"\n💼 Coherencia de negocio:")
        print(f"  Segmentos únicos identificados: {unique_segments}")
        business_quality = "✅ Muy coherente" if unique_segments >= len(insights)*0.8 else "⚠️ Moderadamente coherente" if unique_segments >= len(insights)*0.6 else "❌ Poco coherente"
        print(f"  Evaluación: {business_quality}")
        
        # Evaluación general
        print(f"\n🏆 EVALUACIÓN GENERAL:")
        qualities = [size_quality, sep_quality, business_quality]
        excellent_count = sum([1 for q in qualities if "✅" in q])
        warning_count = sum([1 for q in qualities if "⚠️" in q])
        
        if excellent_count >= 2:
            overall = "✅ CLUSTERS DE ALTA CALIDAD"
        elif warning_count + excellent_count >= 2:
            overall = "⚠️ CLUSTERS DE CALIDAD ACEPTABLE - Considerar ajustes"
        else:
            overall = "❌ CLUSTERS DE BAJA CALIDAD - Se recomienda reconfigurar"
            
        print(f"  {overall}")
        
        return {
            'size_quality': size_quality,
            'separation_quality': sep_quality,
            'business_quality': business_quality,
            'overall_quality': overall
        }


#=========================================== EJEMPLO DE USO ==========================================#
# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar análisis
    analyzer = RFMClusteringAnalysis()
    
    #Ejecutar todos los algoritmos
    results = analyzer.run_all_algorithms()
    
    # Ver todos los algoritmos
    analyzer.plot_all_algorithms_grid(method='tsne')

    #Crear summary report
    summary_report = analyzer.create_summary_report()

    # Mostrar comparación
    print("📊 COMPARACIÓN DE RESULTADOS:")
    print("=" * 80)
    print(results)
    
    # Visualizar métricas
    analyzer.plot_comparison_metrics()
    
    # Obtener mejor algoritmo
    best = analyzer.get_best_algorithm()
    
    # Guardar resultados
    filename = analyzer.save_results_comparison()
    print(f"\n📁 Todos los resultados guardados en: {filename}")


    #==================================== ANÁLISIS DE CLUSTERS ==========================================#

    #analyzer.generate_business_insights()
    #analyzer.get_cluster_statistics()
    #analyzer.analyze_cluster_separation()
    #analyzer.validate_cluster_quality()
    #analyzer.plot_cluster_analysis()