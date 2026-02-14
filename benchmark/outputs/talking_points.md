# DRN vs. AutoML/Classical ML: Key Talking Points

*Generated: 2026-02-14 20:30*

## 1. Quantitative Advantages

- **vs. Logistic Regression**: DRN is +22.0pp more accurate (86.5% vs 64.5%)
  - DRN is 7243.2x slower at inference (but more accurate)
- **vs. Decision Tree**: DRN is +36.0pp more accurate (86.5% vs 50.5%)
  - DRN is 19009.6x slower at inference (but more accurate)
- **vs. Random Forest**: DRN is +23.0pp more accurate (86.5% vs 63.5%)
  - DRN is 72.8x slower at inference (but more accurate)
- **vs. SVM (RBF)**: DRN is +28.0pp more accurate (86.5% vs 58.5%)
  - DRN is 132.0x slower at inference (but more accurate)
- **vs. MLP Classifier**: DRN is +21.0pp more accurate (86.5% vs 65.5%)
  - DRN is 3840.4x slower at inference (but more accurate)
- **vs. H2O AutoML**: DRN is +32.5pp more accurate (86.5% vs 54.0%)
  - DRN is 20.5x slower at inference (but more accurate)
- **vs. AutoGluon**: DRN is +19.0pp more accurate (86.5% vs 67.5%)
  - DRN is 4.8x slower at inference (but more accurate)

## 2. Structural Arguments (Why AutoML is Suboptimal)

- **Flattening destroys spatial structure**: AutoML requires 32x21 -> 672D flattening per region (1,344D combined)
  - Loses positional relationships between adjacent channels and frequency bands
  - DRN processes native 2D structure with 1D convolutions over frequency dimension
- **No dual-region modeling**: AutoML treats OB and PCx as a single concatenated vector
  - Cannot learn region-specific feature importance
  - DRN's fusion attention dynamically weights OB vs PCx contributions
- **No domain-specific inductive bias**: AutoML searches generic pipeline space
  - Cannot exploit known neuroscience (e.g., OB-PCx functional hierarchy)
  - DRN's architecture mirrors the dual-region neural circuitry

## 3. Scalability Arguments

- **DRN inference is O(P)**: 9,882,076 FLOPs per sample, constant regardless of training set size
- **SVM inference scales with data**: O(n_sv x d), grows linearly with n
  - At n=50,000: SVM inference ~44x slower than at n=1,140
  - DRN: identical speed regardless of n
- **DRN memory is fixed**: O(P) = 39.5 MB
- **SVM memory is O(n^2)**: kernel matrix grows quadratically
  - At n=100,000: SVM needs ~75 GB for kernel matrix alone

## 4. Interpretability Arguments

- DRN's attention weights provide neuroscientifically meaningful interpretations
- Channel attention reveals discriminative electrode contacts
- Spatial attention highlights important frequency bands
- Gradient-based saliency maps computed on native 2D structure
- AutoML ensembles offer limited feature-level interpretability

## 5. Real-Time Deployment (BMI Context)

- DRN inference: 25.960 ms per sample (well within real-time)
- Constant-time guarantee critical for brain-machine interface applications
- GPU-acceleratable parallel operations
- SVM/RF inference grows with training data, unsuitable for deployed systems

## 6. Summary Table

| Method | Accuracy | Infer (ms) | Memory | Scales? |
|--------|----------|------------|--------|---------|
| Logistic Regression | 64.5% | 0.004 | 8.2 MB | No |
| Decision Tree | 50.5% | 0.001 | 0.0 MB | Yes |
| Random Forest | 63.5% | 0.356 | 0.9 MB | Yes |
| SVM (RBF) | 58.5% | 0.197 | 7.8 MB | Yes (O(n)) |
| MLP Classifier | 65.5% | 0.007 | 32.6 MB | No |
| H2O AutoML | 54.0% | 1.265 | 1.8 MB | Yes |
| AutoGluon | 67.5% | 5.399 | 47.8 MB | Yes |
| **DRN (Ours)** | **86.5%** | **25.960** | **39.5 MB** | No |