# Módulo de Comparación con Métodos Tradicionales para RFM Analytics
# Este módulo extiende el sistema principal con capacidades de benchmarking

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TraditionalRFMComparator:
    """
    Clase para comparar métodos tradicionales con el sistema adaptativo de RL
    """
    
    def __init__(self, df_customers, adaptive_system=None):
        self.df_customers = df_customers.copy()
        self.adaptive_system = adaptive_system
        self.comparison_results = {}
        self.statistical_tests = {}
        self.feature_importance = {}
        
        # Preparar datos base
        self.prepare_base_data()
        
    def prepare_base_data(self):
        """Preparar datos base para comparaciones"""
        # Calcular scores RFM si no existen
        if 'R_Score' not in self.df_customers.columns:
            self.df_customers['R_Score'] = pd.qcut(self.df_customers['Recencia'], 5, labels=[5,4,3,2,1])
            self.df_customers['F_Score'] = pd.qcut(self.df_customers['Frequencia'].rank(method='first'), 5, labels=[1,2,3,4,5])
            self.df_customers['M_Score'] = pd.qcut(self.df_customers['Valor Monetario'], 5, labels=[1,2,3,4,5])
        
        # Crear features adicionales
        self.df_customers['RFM_Score'] = (
            self.df_customers['R_Score'].astype(int) * 100 + 
            self.df_customers['F_Score'].astype(int) * 10 + 
            self.df_customers['M_Score'].astype(int)
        )
        
        # Simular métricas adicionales si no existen
        if 'CLV' not in self.df_customers.columns:
            self.df_customers['CLV'] = self.df_customers['Valor Monetario'] * np.random.uniform(1.2, 2.5, len(self.df_customers))
        
        # Crear target variables simuladas para evaluación
        self.df_customers['churn_probability'] = np.random.beta(2, 5, len(self.df_customers))
        self.df_customers['satisfaction_score'] = np.random.beta(5, 2, len(self.df_customers))
        self.df_customers['response_rate'] = np.random.beta(3, 4, len(self.df_customers))
        
    def traditional_rfm_segments(self):
        """Implementar segmentación RFM tradicional (11 segmentos)"""
        segments = []
        
        for _, row in self.df_customers.iterrows():
            r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
            
            if r >= 4 and f >= 4 and m >= 4:
                segment = 'Champions'
            elif r >= 2 and f >= 3 and m >= 3:
                segment = 'Loyal Customers'
            elif r >= 3 and f <= 3 and m <= 3:
                segment = 'Potential Loyalists'
            elif r >= 4 and f <= 1 and m <= 1:
                segment = 'New Customers'
            elif r >= 3 and f <= 2 and m >= 3:
                segment = 'Promising'
            elif r <= 2 and f >= 2 and m >= 2:
                segment = 'Customers Needing Attention'
            elif r <= 2 and f >= 3 and m >= 3:
                segment = 'About to Sleep'
            elif r <= 2 and f >= 4 and m >= 4:
                segment = 'At Risk'
            elif r <= 1 and f >= 4 and m >= 4:
                segment = 'Cannot Lose Them'
            elif r <= 2 and f <= 2 and m <= 2:
                segment = 'Hibernating'
            else:
                segment = 'Lost'
                
            segments.append(segment)
        
        return segments
    
    def kmeans_segmentation(self, n_clusters=8):
        """Segmentación usando K-Means"""
        features = ['R_Score', 'F_Score', 'M_Score']
        X = self.df_customers[features].astype(float)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calcular métricas de calidad
        silhouette_avg = silhouette_score(X_scaled, clusters)
        inertia = kmeans.inertia_
        
        return clusters, {'silhouette_score': silhouette_avg, 'inertia': inertia, 'model': kmeans}
    
    def dbscan_segmentation(self, eps=0.5, min_samples=5):
        """Segmentación usando DBSCAN"""
        features = ['R_Score', 'F_Score', 'M_Score']
        X = self.df_customers[features].astype(float)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        silhouette_avg = silhouette_score(X_scaled, clusters) if n_clusters > 1 else -1
        
        return clusters, {
            'n_clusters': n_clusters, 
            'n_noise': n_noise, 
            'silhouette_score': silhouette_avg,
            'model': dbscan
        }
    
    def hierarchical_clustering(self, n_clusters=8):
        """Segmentación jerárquica"""
        features = ['R_Score', 'F_Score', 'M_Score']
        X = self.df_customers[features].astype(float)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = hierarchical.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        return clusters, {'silhouette_score': silhouette_avg, 'model': hierarchical}
    
    def decision_tree_segmentation(self):
        """Segmentación usando árboles de decisión supervisados"""
        features = ['R_Score', 'F_Score', 'M_Score', 'Valor Monetario', 'Frequencia', 'Recencia']
        X = self.df_customers[features]
        
        # Crear target basado en quintiles de CLV
        y = pd.qcut(self.df_customers['CLV'], 5, labels=['Low', 'Below_Avg', 'Average', 'Above_Avg', 'High'])
        
        dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
        dt.fit(X, y)
        
        segments = dt.predict(X)
        cv_scores = cross_val_score(dt, X, y, cv=5)
        
        # Feature importance
        feature_importance = dict(zip(features, dt.feature_importances_))
        
        return segments, {
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'model': dt
        }
    
    def random_forest_segmentation(self):
        """Segmentación usando Random Forest"""
        features = ['R_Score', 'F_Score', 'M_Score', 'Valor Monetario', 'Frequencia', 'Recencia']
        X = self.df_customers[features]
        
        # Target basado en CLV y churn risk combinados
        clv_quintile = pd.qcut(self.df_customers['CLV'], 3, labels=[0, 1, 2])
        churn_tertile = pd.qcut(self.df_customers['churn_probability'], 3, labels=[0, 1, 2])
        y = clv_quintile.astype(int) * 3 + churn_tertile.astype(int)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        rf.fit(X, y)
        
        segments = rf.predict(X)
        cv_scores = cross_val_score(rf, X, y, cv=5)
        
        feature_importance = dict(zip(features, rf.feature_importances_))
        
        return segments, {
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'model': rf
        }
    
    def logistic_regression_segmentation(self):
        """Segmentación probabilística usando regresión logística"""
        features = ['R_Score', 'F_Score', 'M_Score', 'Valor Monetario', 'Frequencia', 'Recencia']
        X = self.df_customers[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Target binario: high value vs low value customers
        median_clv = self.df_customers['CLV'].median()
        y = (self.df_customers['CLV'] > median_clv).astype(int)
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)
        
        # Probabilidades como scores de segmentación
        probabilities = lr.predict_proba(X_scaled)[:, 1]
        segments = pd.qcut(probabilities, 4, labels=['Low_Prob', 'Med_Low', 'Med_High', 'High_Prob'])
        
        cv_scores = cross_val_score(lr, X_scaled, y, cv=5)
        
        return segments, {
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'probabilities': probabilities,
            'model': lr,
            'scaler': scaler
        }
    
    def evaluate_segmentation_quality(self, segments, method_name):
        """Evaluar calidad de segmentación"""
        segments_array = np.array(segments)
        
        # Métricas internas
        features = ['R_Score', 'F_Score', 'M_Score']
        X = self.df_customers[features].astype(float)
        
        # Silhouette score
        if len(np.unique(segments_array)) > 1:
            silhouette_avg = silhouette_score(X, segments_array)
        else:
            silhouette_avg = -1
        
        # Métricas de negocio
        segment_df = self.df_customers.copy()
        segment_df['Segment'] = segments
        
        # CLV promedio por segmento
        avg_clv_by_segment = segment_df.groupby('Segment')['CLV'].mean()
        clv_variance = avg_clv_by_segment.var()
        
        # Distribución de segmentos
        segment_distribution = segment_df['Segment'].value_counts(normalize=True)
        
        # Balance de segmentos (entropía)
        segment_entropy = stats.entropy(segment_distribution)
        
        # ROI potencial por segmento
        segment_stats = segment_df.groupby('Segment').agg({
            'CLV': ['mean', 'std', 'count'],
            'churn_probability': 'mean',
            'satisfaction_score': 'mean',
            'response_rate': 'mean'
        }).round(3)
        
        return {
            'silhouette_score': silhouette_avg,
            'clv_variance': clv_variance,
            'segment_entropy': segment_entropy,
            'segment_distribution': segment_distribution.to_dict(),
            'segment_stats': segment_stats,
            'n_segments': len(np.unique(segments_array))
        }
    
    def compare_all_methods(self):
        """Comparar todos los métodos de segmentación"""
        print("🔄 Ejecutando comparación exhaustiva de métodos...")
        
        methods = {
            'Traditional_RFM': self.traditional_rfm_segments,
            'KMeans': lambda: self.kmeans_segmentation(n_clusters=8),
            'DBSCAN': lambda: self.dbscan_segmentation(eps=0.8, min_samples=5),
            'Hierarchical': lambda: self.hierarchical_clustering(n_clusters=8),
            'Decision_Tree': self.decision_tree_segmentation,
            'Random_Forest': self.random_forest_segmentation,
            'Logistic_Regression': self.logistic_regression_segmentation
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"   Evaluando {method_name}...")
            
            try:
                if method_name == 'Traditional_RFM':
                    segments = method_func()
                    method_metrics = {}
                else:
                    segments, method_metrics = method_func()
                
                # Evaluar calidad
                quality_metrics = self.evaluate_segmentation_quality(segments, method_name)
                
                # Simular métricas de negocio
                business_metrics = self.simulate_business_performance(segments, method_name)
                
                results[method_name] = {
                    'segments': segments,
                    'method_metrics': method_metrics,
                    'quality_metrics': quality_metrics,
                    'business_metrics': business_metrics
                }
                
            except Exception as e:
                print(f"   ❌ Error en {method_name}: {str(e)}")
                continue
        
        self.comparison_results = results
        return results
    
    def simulate_business_performance(self, segments, method_name, simulation_days=90):
        """Simular rendimiento de negocio para cada método"""
        segment_df = self.df_customers.copy()
        segment_df['Segment'] = segments
        
        # Definir estrategias por método
        strategy_map = self.get_strategy_mapping(method_name)
        
        total_revenue = 0
        total_cost = 0
        churn_prevented = 0
        satisfaction_improvement = 0
        
        # Simular para cada segmento
        for segment in segment_df['Segment'].unique():
            segment_customers = segment_df[segment_df['Segment'] == segment]
            n_customers = len(segment_customers)
            
            # Obtener estrategia para este segmento
            strategy = strategy_map.get(str(segment), 'no_action')
            
            # Simular efectos durante el período
            segment_revenue, segment_cost, segment_churn_prevention, segment_satisfaction = \
                self.simulate_strategy_effects(segment_customers, strategy, simulation_days)
            
            total_revenue += segment_revenue
            total_cost += segment_cost
            churn_prevented += segment_churn_prevention
            satisfaction_improvement += segment_satisfaction
        
        # Calcular métricas finales
        roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
        avg_churn_prevention = churn_prevented / len(segment_df)
        avg_satisfaction_improvement = satisfaction_improvement / len(segment_df)
        
        return {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'roi': roi,
            'churn_prevention_rate': avg_churn_prevention,
            'satisfaction_improvement': avg_satisfaction_improvement,
            'revenue_per_customer': total_revenue / len(segment_df),
            'cost_per_customer': total_cost / len(segment_df)
        }
    
    def get_strategy_mapping(self, method_name):
        """Definir estrategias por método y segmento"""
        if method_name == 'Traditional_RFM':
            return {
                'Champions': 'premium_offer',
                'Loyal Customers': 'cross_sell',
                'Potential Loyalists': 'retention_campaign',
                'New Customers': 'welcome_program',
                'Promising': 'engagement_boost',
                'Customers Needing Attention': 'attention_campaign',
                'About to Sleep': 'wake_up_campaign',
                'At Risk': 'churn_prevention',
                'Cannot Lose Them': 'vip_treatment',
                'Hibernating': 'reactivation',
                'Lost': 'winback_campaign'
            }
        else:
            # Estrategias genéricas para otros métodos
            return {
                '0': 'discount_offer', '1': 'premium_offer', '2': 'retention_campaign',
                '3': 'cross_sell', '4': 'churn_prevention', '5': 'engagement_boost',
                '6': 'vip_treatment', '7': 'reactivation', '8': 'winback_campaign',
                'Low': 'discount_offer', 'Below_Avg': 'retention_campaign',
                'Average': 'cross_sell', 'Above_Avg': 'premium_offer', 'High': 'vip_treatment',
                'Low_Prob': 'discount_offer', 'Med_Low': 'retention_campaign',
                'Med_High': 'cross_sell', 'High_Prob': 'premium_offer'
            }
    
    def simulate_strategy_effects(self, customers, strategy, days):
        """Simular efectos de una estrategia específica"""
        n_customers = len(customers)
        
        # Definir efectos por estrategia
        strategy_effects = {
            'premium_offer': {'revenue_mult': 2.5, 'cost_per_customer': 50, 'churn_reduction': 0.15, 'satisfaction_boost': 0.2},
            'discount_offer': {'revenue_mult': 1.8, 'cost_per_customer': 30, 'churn_reduction': 0.1, 'satisfaction_boost': 0.1},
            'retention_campaign': {'revenue_mult': 1.2, 'cost_per_customer': 25, 'churn_reduction': 0.25, 'satisfaction_boost': 0.15},
            'cross_sell': {'revenue_mult': 2.0, 'cost_per_customer': 35, 'churn_reduction': 0.05, 'satisfaction_boost': 0.08},
            'churn_prevention': {'revenue_mult': 1.1, 'cost_per_customer': 40, 'churn_reduction': 0.3, 'satisfaction_boost': 0.12},
            'vip_treatment': {'revenue_mult': 3.0, 'cost_per_customer': 80, 'churn_reduction': 0.2, 'satisfaction_boost': 0.25},
            'no_action': {'revenue_mult': 1.0, 'cost_per_customer': 0, 'churn_reduction': 0, 'satisfaction_boost': 0}
        }
        
        effects = strategy_effects.get(strategy, strategy_effects['no_action'])
        
        # Calcular métricas
        base_revenue = customers['CLV'].sum() / 365 * days  # Revenue durante el período
        total_revenue = base_revenue * effects['revenue_mult']
        total_cost = n_customers * effects['cost_per_customer']
        
        churn_prevention = n_customers * effects['churn_reduction']
        satisfaction_improvement = n_customers * effects['satisfaction_boost']
        
        return total_revenue, total_cost, churn_prevention, satisfaction_improvement
    
    def statistical_significance_tests(self):
        """Realizar pruebas de significancia estadística"""
        if not self.comparison_results:
            print("❌ Primero ejecuta compare_all_methods()")
            return
        
        print("🔬 Realizando pruebas de significancia estadística...")
        
        # Extraer métricas para comparación
        methods = list(self.comparison_results.keys())
        roi_values = [self.comparison_results[method]['business_metrics']['roi'] for method in methods]
        silhouette_values = [self.comparison_results[method]['quality_metrics']['silhouette_score'] for method in methods]
        
        # Prueba ANOVA para ROI
        from scipy.stats import f_oneway, kruskal
        
        # Crear grupos de datos simulados para cada método (necesario para ANOVA)
        roi_groups = []
        for method in methods:
            # Simular variabilidad en las métricas
            base_roi = self.comparison_results[method]['business_metrics']['roi']
            roi_group = np.random.normal(base_roi, abs(base_roi * 0.1), 30)  # 30 simulaciones por método
            roi_groups.append(roi_group)
        
        # ANOVA para ROI
        f_stat, p_value_anova = f_oneway(*roi_groups)
        
        # Kruskal-Wallis (no paramétrica)
        h_stat, p_value_kruskal = kruskal(*roi_groups)
        
        # Pruebas pareadas (t-test entre métodos)
        from scipy.stats import ttest_ind
        pairwise_tests = {}
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                t_stat, p_val = ttest_ind(roi_groups[i], roi_groups[j])
                pairwise_tests[f"{method1}_vs_{method2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
        
        self.statistical_tests = {
            'anova': {'f_statistic': f_stat, 'p_value': p_value_anova, 'significant': p_value_anova < 0.05},
            'kruskal_wallis': {'h_statistic': h_stat, 'p_value': p_value_kruskal, 'significant': p_value_kruskal < 0.05},
            'pairwise_tests': pairwise_tests
        }
        
        return self.statistical_tests
    
    def generate_comprehensive_report(self):
        """Generar reporte comprehensive de comparación"""
        if not self.comparison_results:
            print("❌ Primero ejecuta compare_all_methods()")
            return
        
        print("=" * 80)
        print("📊 REPORTE COMPREHSIVO DE COMPARACIÓN DE MÉTODOS")
        print("=" * 80)
        
        # Crear DataFrame de resultados
        report_data = []
        for method, results in self.comparison_results.items():
            report_data.append({
                'Método': method,
                'ROI': results['business_metrics']['roi'],
                'Ingresos_Totales': results['business_metrics']['total_revenue'],
                'Silhouette_Score': results['quality_metrics']['silhouette_score'],
                'N_Segmentos': results['quality_metrics']['n_segments'],
                'Prevención_Churn': results['business_metrics']['churn_prevention_rate'],
                'Mejora_Satisfacción': results['business_metrics']['satisfaction_improvement'],
                'Ingreso_por_Cliente': results['business_metrics']['revenue_per_customer'],
                'Costo_por_Cliente': results['business_metrics']['cost_per_customer']
            })
        
        df_report = pd.DataFrame(report_data)
        df_report = df_report.sort_values('ROI', ascending=False)
        
        print("\n🏆 RANKING GENERAL (por ROI):")
        print(df_report[['Método', 'ROI', 'Silhouette_Score', 'N_Segmentos']].round(3).to_string(index=False))
        
        # Mejor método por cada métrica
        print(f"\n🥇 MEJORES MÉTODOS POR MÉTRICA:")
        print(f"   • Mayor ROI: {df_report.loc[df_report['ROI'].idxmax(), 'Método']} ({df_report['ROI'].max():.3f})")
        print(f"   • Mejor Calidad Segmentación: {df_report.loc[df_report['Silhouette_Score'].idxmax(), 'Método']} ({df_report['Silhouette_Score'].max():.3f})")
        print(f"   • Mayor Prevención Churn: {df_report.loc[df_report['Prevención_Churn'].idxmax(), 'Método']} ({df_report['Prevención_Churn'].max():.3f})")
        
        # Análisis de trade-offs
        print(f"\n⚖️ ANÁLISIS DE TRADE-OFFS:")
        high_roi_methods = df_report[df_report['ROI'] > df_report['ROI'].median()]['Método'].tolist()
        high_quality_methods = df_report[df_report['Silhouette_Score'] > df_report['Silhouette_Score'].median()]['Método'].tolist()
        
        balanced_methods = list(set(high_roi_methods) & set(high_quality_methods))
        print(f"   • Métodos balanceados (alto ROI + alta calidad): {balanced_methods}")
        
        return df_report
    
    def plot_comprehensive_comparison(self):
        """Crear visualizaciones comprehensivas de la comparación"""
        if not self.comparison_results:
            print("❌ Primero ejecuta compare_all_methods()")
            return
        
        # Preparar datos
        methods = list(self.comparison_results.keys())
        roi_values = [self.comparison_results[method]['business_metrics']['roi'] for method in methods]
        silhouette_values = [self.comparison_results[method]['quality_metrics']['silhouette_score'] for method in methods]
        revenue_values = [self.comparison_results[method]['business_metrics']['total_revenue'] for method in methods]
        churn_prevention = [self.comparison_results[method]['business_metrics']['churn_prevention_rate'] for method in methods]
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. ROI Comparison
        ax1 = plt.subplot(2, 3, 1)
        bars1 = ax1.bar(methods, roi_values, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
        ax1.set_title('Comparación de ROI por Método')
        ax1.set_ylabel('ROI')
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en barras
        for bar, value in zip(bars1, roi_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. Silhouette Score
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(methods, silhouette_values, color=plt.cm.plasma(np.linspace(0, 1, len(methods))))
        ax2.set_title('Calidad de Segmentación (Silhouette Score)')
        ax2.set_ylabel('Silhouette Score')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, silhouette_values):
            if value > -1:  # Solo mostrar valores válidos
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Revenue vs Cost Scatter
        ax3 = plt.subplot(2, 3, 3)
        costs = [self.comparison_results[method]['business_metrics']['total_cost'] for method in methods]
        scatter = ax3.scatter(costs, revenue_values, s=100, c=roi_values, cmap='viridis', alpha=0.7)
        
        for i, method in enumerate(methods):
            ax3.annotate(method, (costs[i], revenue_values[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Costo Total')
        ax3.set_ylabel('Ingresos Totales')
        ax3.set_title('Ingresos vs Costos (color = ROI)')
        plt.colorbar(scatter, ax=ax3, label='ROI')
        
        # 4. Radar Chart de Métricas Múltiples
        ax4 = plt.subplot(2, 3, 4, projection='polar')
        
        categories = ['ROI', 'Silhouette', 'Churn_Prev', 'Revenue']
        
        # Normalizar datos para radar chart
        roi_norm = np.array(roi_values) / max(roi_values) if max(roi_values) > 0 else np.zeros_like(roi_values)
        silh_norm = np.array([max(0, s) for s in silhouette_values]) / max([max(0, s) for s in silhouette_values]) if max(silhouette_values) > 0 else np.zeros_