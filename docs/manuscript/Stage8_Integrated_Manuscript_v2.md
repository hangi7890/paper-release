# Data-Efficient Cross-Building Transfer Learning for 24-Hour Electricity Forecasting: Target Adaptation Benefits and Failure Conditions

## Abstract

Cross-building transfer learning can support electricity forecasting when target-building data are limited, but adaptation gains may not translate into superiority over simple forecasting rules. This study evaluates data-efficient target adaptation for direct 24-hour electricity forecasting using hourly records from the Building Data Genome Project 2. A compact LSTM was pretrained on source buildings and evaluated without target adaptation and after output-head-only adaptation using 1, 3, 7, 14, and 28 consecutive days of target observations. The final evaluation included 453 quality-controlled buildings across seven site-disjoint final evaluation sites under a fixed 28-day test protocol. Equal-weighted site MAE decreased from 40.65 in the source-only condition to 24.17 after 28 days of adaptation, an approximately 40.5% reduction. The site-equal F1 event rate decreased from approximately 60.3% at the 1-day budget to 5.2% at the 28-day budget. At 28 days, the site-equal source-improvement rate was 94.8%, whereas the corresponding rate for outperforming both previous-day and weekly-same-hour baselines was 26.8%. Auxiliary Jetson Orin Nano testing reproduced reference FP32 predictions within predefined tolerances and achieved sub-millisecond typical batch-1 inference latency. These results show that additional target data improve adaptation accuracy and reliability, while source-model improvement alone is insufficient evidence of absolute forecasting superiority.

**Keywords:** cross-building transfer learning; building electricity forecasting; target adaptation; data efficiency; adaptation reliability; Building Data Genome Project 2 (BDG2)

# 1. Introduction

The increasing availability of smart meters, Internet of Things (IoT) devices, and edge-computing platforms has expanded the role of data-driven methods in building energy management. Beyond real-time monitoring, forecasting can provide advance information about expected electricity use and thereby support more anticipatory operation and analysis. In our previous study, an artificial intelligence of things (AIoT)-based energy digital-twin platform was implemented for a laboratory environment, integrating environmental and electrical sensing, occupancy inference, edge processing, data services, digital-twin visualization, and rule-based operating modes [1]. However, the quantitative evaluation in that study primarily concerned occupancy inference and edge-processing performance rather than next-24-hour electricity forecasting. The present study therefore extends the earlier system-level motivation toward predictive modeling without treating the previous platform as evidence that cross-building electricity forecasting or predictive control had already been demonstrated.

A major obstacle to data-driven electricity forecasting is the amount of historical data required for a newly monitored building. Models trained independently for each target building can be difficult to establish when only a short operating history is available. Cross-building transfer learning offers an alternative by reusing representations or parameters learned from data-rich buildings and subsequently applying or adapting them to another building. Earlier studies demonstrated that information from other buildings can improve forecasting under limited target data and investigated transfer strategies based on seasonal adjustment, deep neural networks, and statistical evaluation [2–4]. These studies established the feasibility of cross-building knowledge transfer, while also showing that forecasting performance depends on the amount and characteristics of the data available in the source and target domains.

Recent research has substantially expanded this problem beyond determining whether transfer learning can work. Spencer et al. compared multiple transfer-learning strategies using Transformer-based forecasting architectures and showed that the effectiveness of transfer depends on factors such as the availability of target data, feature space, and temporal alignment [5]. Liu et al. proposed a few-shot cross-building framework for previously unseen buildings with limited target-domain observations [6]. Wu et al. further quantified the effects of source- and target-data volumes across multiple deep transfer strategies [7], while Lei et al. addressed inter-building domain adaptation explicitly under data-scarcity conditions [8]. Taken together, these studies indicate that the amount of target-building data should be treated as an experimental variable rather than simply classifying transfer as either zero-shot or fine-tuned.

An equally important issue is that adaptation is not uniformly beneficial. Differences among buildings in load magnitude, operating schedules, temporal regularity, climate, and other latent characteristics can make a source-trained representation unsuitable for a particular target. Recent work has therefore begun to examine source selection, transferability, robustness, and negative transfer rather than reporting only average performance improvements. Choi et al. investigated similarity-guided transfer across anonymized building time series [9], and Bian et al. demonstrated that cross-domain transfer can exhibit substantial model-dependent variability and negative transfer under distribution shifts [10]. Consequently, an average reduction in forecasting error is insufficient to characterize adaptation reliability. A deployment-oriented analysis should also identify how often adaptation worsens performance, whether its direction is stable across repeated training runs, and how such failures change as additional target observations become available.

A second distinction is required between **relative transfer improvement** and **absolute forecasting competitiveness**. An adapted model may improve substantially relative to the same model before target adaptation and still perform worse than a simple forecasting rule that exploits strong temporal persistence or seasonality. General forecasting practice emphasizes the importance of meaningful benchmark models when interpreting predictive performance [11]. This distinction is particularly relevant for building electricity series, where daily and weekly recurrence can provide strong reference forecasts. Therefore, the present study does not define successful transfer solely as an improvement over the source-only model. It separately evaluates whether an adapted model also outperforms previous-day and weekly-same-hour temporal baselines. This separation provides a more conservative assessment of whether adaptation produces an internally improved transferred model or an operationally competitive forecast.

Practical deployment also introduces a computational dimension. Recent building-energy studies have combined transfer learning with edge computing to reduce dependence on centralized processing and to support forecasting close to the data source. For example, Iqbal and Adnan integrated transfer learning and edge computing for short-term residential and commercial building load forecasting [12]. Such work demonstrates that model accuracy and hardware feasibility can be considered together, but these are distinct questions: successful transfer does not by itself demonstrate that the forecasting architecture is suitable for deployment on compact edge hardware. Accordingly, edge inference in the present study is treated as an auxiliary engineering validation rather than as an additional forecasting hypothesis or as evidence of improved forecasting accuracy.

Against this background, this study investigates **data-efficient cross-building transfer for direct 24-hour electricity forecasting**, using hourly electricity-meter records from the Building Data Genome Project 2 (BDG2) [13]. Rather than introducing another forecasting architecture, the study deliberately uses a compact LSTM-based model and concentrates on the behavior of target adaptation under a fixed evaluation protocol. A source model is evaluated without target adaptation and after head-only adaptation using 1, 3, 7, 14, or 28 consecutive days of target-building data. Each target is evaluated on a fixed 28-day test interval, and the same test definition is retained across adaptation budgets. Development and final evaluation sites are separated, and the final analysis aggregates buildings within sites before giving equal weight to sites. The study addresses three research questions: **(RQ1)** how does forecasting error change as the available target-adaptation data increase; **(RQ2)** how do adaptation-induced degradation, baseline underperformance, and seed-dependent directional instability change as more target-building data become available; and **(RQ3)** to what extent does improvement over the source-only model translate into superiority over strong temporal baselines? The study additionally performs a leakage-controlled exploratory analysis of building characteristics and an independent FP32 deployment validation of the frozen inference architecture on an NVIDIA Jetson Orin Nano. The use of BDG2 as the common empirical basis also permits the analysis to focus on cross-building variation without introducing additional dataset-specific target definitions. The BDG2 dataset itself comprises real building meter data and accompanying metadata collected for large-scale building-energy analytics [13].

The main contribution is therefore not a claim that transfer learning, few-shot adaptation, or edge forecasting is new in itself. Instead, the study provides a controlled analysis of **how much target data are required for adaptation benefits to emerge and how those benefits can fail**. Specifically, it (i) quantifies a predefined 0-, 1-, 3-, 7-, 14-, and 28-day adaptation curve under a common fixed-test protocol; (ii) separates improvement relative to the source-only model from superiority over simple temporal baselines; (iii) evaluates predefined failure modes, including adaptation-induced degradation, baseline underperformance, and seed-dependent directional instability; and (iv) supplements the forecasting analysis with hardware-level numerical-consistency and inference-feasibility measurements on an edge device. This framing complements existing work on transfer architectures and source-target adaptation by placing greater emphasis on data efficiency, failure conditions, and the distinction between relative and absolute forecasting success.

The final evaluation involved 453 buildings across seven sites. Increasing the adaptation budget substantially reduced the site-equal error relative to the source-only model: at 28 days, the site-equal MAE decreased from 40.65 to 24.17, corresponding to an approximately 40.5% reduction. Nevertheless, the 28-day adapted model remained above the previous-day and weekly-same-hour baselines in site-equal MAE, and the distinction was also visible in the site-equal rates: the source-improvement rate was approximately 94.8%, whereas the rate for outperforming both temporal baselines was approximately 26.8%. Thus, the principal finding is that **greater target adaptation data can substantially mitigate cross-building transfer error, but relative transfer improvement does not necessarily imply absolute superiority over strong simple forecasts**. The auxiliary deployment experiment further showed that the frozen compact inference architecture could reproduce the reference predictions on a Jetson Orin Nano within the predefined FP32 tolerances, supporting feasibility under the tested hardware and runtime configuration without implying general real-time or cross-device energy-efficiency superiority.

# 2. Materials and Methods

## 2.1. Study Design and Experimental Overview

This study evaluated data-efficient cross-building transfer learning for direct 24-hour electricity forecasting under limited target-building data. The experimental design separated model development from a subsequent fixed final evaluation. A compact source model was first trained using a predefined set of source buildings. Target adaptation was then evaluated using fixed data budgets of 0, 1, 3, 7, 14, and 28 days. The 0-day condition represented source-only inference without any target-label-based parameter update, whereas the remaining conditions applied the same predefined head-only adaptation procedure using progressively larger amounts of chronological target-building data.

For every target building, all adaptation conditions were evaluated against the same fixed 28-day test interval. The test period consisted of 28 non-overlapping daily forecast origins, each producing a direct 24-hour forecast, resulting in 672 evaluated hourly targets per building. Adaptation and test targets were temporally disjoint, and the target test period was not shifted according to model performance or adaptation budget. The final protocol further prohibited post-result changes to the evaluated buildings, time periods, random seeds, learning rate, epoch limits, performance metrics, failure thresholds, or representative-case selection rules.

The study used three fixed random seeds (20260924, 20260925, and 20260926) for repeated model adaptation. Seed-level runs were treated as repeated model realizations rather than independent buildings. Buildings were nested within sites; therefore, inferential summaries were constructed first at the building level and subsequently at the site level rather than treating individual hourly observations or repeated seeds as independent samples.

**[Figure 1 about here: Overall experimental framework]**

**Figure 1.** Overview of the cross-building forecasting and target-adaptation protocol. A source-trained LSTM is evaluated without target adaptation and after 1-, 3-, 7-, 14-, and 28-day head-only adaptation. All conditions share the same fixed 28-day target test interval.

---

## 2.2. Dataset and Building Cohort

The experiments were conducted using hourly electricity-meter records from the Building Data Genome Project 2 (BDG2) [13]. BDG2 provides large-scale meter data and building metadata collected from multiple sites and therefore enables evaluation of transfer across heterogeneous buildings while retaining a common dataset origin.

The raw electricity dataset contained 17,544 hourly timestamps and 1,578 building-meter columns covering 2016–2017. Building eligibility was determined using a fixed quality-control procedure before final forecasting evaluation. Candidate records were required to contain numeric electricity values, not be entirely zero, contain at least one continuous valid segment of 90 days, contain no negative values, contain no continuous zero-valued period of 30 days or longer, and have no more than 20% missing values. Fixed building quotas and deterministic ranking rules were used for the source and development cohorts.

The source cohort consisted of 20 buildings from four sites, with five buildings selected from each source site. A separate development cohort contained 12 buildings from six sites, with two buildings per site. These cohorts were used for model development, adaptation-protocol design, and failure-analysis formulation before the final test.

The predefined final evaluation proposal initially contained 510 buildings across nine sites. Application of the frozen final eligibility rule excluded 57 buildings, all due to continuous zero-valued periods of at least 30 days. The resulting final evaluation cohort contained **453 buildings across seven sites**: Bobcat (22), Bull (111), Eagle (94), Gator (63), Hog (135), Moose (10), and Swan (18). Mouse and Shrew contained no retained eligible buildings. No replacement buildings or shifted evaluation periods were selected after these exclusions.

The physical metering boundary of every individual BDG2 electricity series could not be independently demonstrated to be identical across all buildings. Accordingly, the study refers conservatively to the analyzed variables as **hourly electricity-meter records** or **electricity forecasting targets**, rather than assuming that every series represents an identically defined whole-building electricity-demand boundary. This limitation is retained when interpreting differences in absolute load magnitude across buildings.

**[Table 1 about here: Experimental cohorts and final evaluation sites]**

---

## 2.3. Forecasting Task and Input Representation

For a target building \(b\), the task was to predict the electricity-meter values for the next 24 hourly intervals from information available at the forecast issue time. The forecasting model used an input sequence of length 24 with six variables per timestep and generated a 24-dimensional direct multi-horizon output.

The input consisted of the previous 24 hourly electricity observations together with deterministic calendar variables associated with the target forecast horizon. The calendar representation contained sine and cosine encodings of hour of day, sine and cosine encodings of day of week, and a weekend indicator. No future measured electricity or other future-observed covariates were used as model inputs.

For a forecast origin \(F\), the historical electricity input covered \(F-24\) through \(F-1\), while the forecast targets covered \(F\) through \(F+23\). Calendar variables for the target horizon were deterministic and therefore known at forecast issue time. All evaluated models and adaptation budgets for a given building used identical forecast origins and identical target timestamps. Final integrity checks confirmed that the combination of forecast origin, target timestamp, forecast horizon, and observed target value was identical across conditions and that no overlap existed between adaptation labels and test labels.

The numerical input tensor therefore had shape

\[
\mathbf{X}\in \mathbb{R}^{24\times6},
\]

and the model produced

\[
\hat{\mathbf{y}}\in \mathbb{R}^{24}.
\]

No autoregressive use of predictions was required within the 24-hour horizon.

---

## 2.4. Source Model

A compact long short-term memory (LSTM) network was used as the forecasting model. The network consisted of a single LSTM layer with a hidden size of 32 followed by a linear output layer producing the 24 forecast horizons simultaneously. The model contained 5,912 trainable parameters.

Source-model training used 218,206 supervised windows, with an additional 12,980 chronologically separated source-validation windows. Model selection and early stopping were based exclusively on the source-validation data; target-building test observations were not used for normalization, early stopping, checkpoint selection, or hyperparameter adjustment.

The source model was optimized using Adam with a learning rate of 0.001 and a batch size of 256 for a maximum of 100 epochs. Early stopping used a patience of 10 epochs. Three prespecified random seeds (20260924, 20260925, and 20260926) were used throughout the experiment. The source-model training was executed on an NVIDIA GeForce RTX 3080.

Electricity values were normalized using statistics calculated exclusively from the fixed source-training inputs and targets. The frozen source normalization mean and population standard deviation were 167.065198 and 354.328670, respectively. These same normalization parameters were retained during target adaptation and final evaluation rather than being recalculated from target test data.

The source model was optimized using the mean squared error (MSE) loss on source-normalized targets.

---

## 2.5. Target Adaptation and Data Budgets

Target adaptation was restricted to the model output head. For every nonzero target-data budget, the pretrained LSTM representation was frozen and only the parameters of the linear 24-output forecasting head were updated. Consequently, target adaptation changed parameter values without altering the model architecture, input representation, output dimensionality, or inference parameter count.

Target adaptation used the same mean squared error loss as source training. Adaptation used Adam with a learning rate of 0.0005 for 50 epochs. The same adaptation procedure was applied to all target buildings, budgets, and three prespecified random seeds. No target-specific validation split or test-based early stopping was used during adaptation. This choice was retained across all budgets so that the 1-day condition, which contained only one complete supervised training window, was evaluated under the same adaptation rule as the larger-data conditions.

The nominal adaptation batch size was 64, with the effective DataLoader batch size defined as the smaller of 64 and the number of available adaptation samples; consequently, the effective batch sizes were 1, 49, 64, 64, and 64 for the 1-, 3-, 7-, 14-, and 28-day budgets, respectively, with incomplete final batches retained.

The target-data budgets were defined as 1, 3, 7, 14, and 28 consecutive days of target-building observations, with the 0-day condition representing the source-only model without target-label-based parameter updates. Because each supervised example required a 24-h historical input followed by a 24-h target horizon and training windows were generated with an hourly stride, the number of available adaptation windows was not equal to the nominal number of adaptation days. The corresponding numbers of adaptation windows were 1, 49, 145, 313, and 649 for the 1-, 3-, 7-, 14-, and 28-day budgets, respectively.

All adaptation budgets were evaluated on the same fixed 28-day test interval for each building. The primary evaluation used 28 non-overlapping daily forecast origins, corresponding to 672 hourly target values per building and seed. Adaptation targets ended before the test targets began, and no target timestamp was shared between the adaptation and test sets. Independent reproducibility checks additionally confirmed that the first forecasted hour always followed the end of the input sequence by exactly one hour.

| Configuration | Source pretraining | Target adaptation |
|---|---:|---:|
| Input length | 24 h | 24 h |
| Forecast horizon | 24 h | 24 h |
| LSTM layers | 1 | Frozen |
| Hidden size | 32 | Frozen |
| Trainable component | Full network | Output head only |
| Loss function | Mean squared error | Mean squared error |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.0005 |
| Maximum epochs | 100 | 50 |
| Batch size | 256 | Nominal maximum 64 |
| Effective batch-size rule | 256 in the completed run | min(64, number of adaptation samples) |
| Effective batch sizes (1/3/7/14/28 d) | — | 1 / 49 / 64 / 64 / 64 |
| Partial final batch | Retained (`drop_last=False`) | Retained (`drop_last=False`) |
| Early stopping | Patience 10 | None |
| Target validation | — | None |
| Seeds | 20260924, 20260925, 20260926 | Same three seeds |
| Parameters at inference | 5,912 | 5,912 |

---

## 2.6. Baseline Forecasts and Evaluation Metrics

Two deterministic temporal baselines were used to distinguish improvement over the transferred source model from improvement over simple forecasting rules. The **previous-day** baseline reused the 24 hourly observations immediately preceding each forecast origin as the next 24-hour prediction. The **weekly-same-hour** baseline used the observation recorded exactly seven days before each target hour. If the historical value required by a baseline was unavailable, it was not replaced using future information or imputation.

Forecasting accuracy was primarily evaluated using mean absolute error (MAE) and root mean square error (RMSE). For each building, seed, and adaptation budget, both metrics were calculated over the common set of 672 hourly targets corresponding to the 28 fixed daily forecast windows. MAE was used as the primary performance measure, while RMSE was reported as a complementary measure that places greater weight on larger forecast errors.

MAE and RMSE are reported in the original electricity-meter units.

For adaptation budget \(d\), improvement relative to the source-only model was quantified as

\[
\Delta \mathrm{MAE}_{\mathrm{source}}(d)
=
\mathrm{MAE}(d)-\mathrm{MAE}(0),
\]

where negative values indicate lower error after target adaptation. The analogous difference was calculated for RMSE. Relative MAE improvement was additionally expressed as the percentage reduction from the source-only MAE.

Performance against the deterministic baselines was evaluated separately. An adapted model was considered better than a baseline only when its MAE was lower than that baseline under the same target test interval. Performance against the previous-day and weekly-same-hour baselines was reported individually, and the intersection in which an adapted model outperformed both baselines was analyzed as a stricter success criterion. This distinction was retained throughout the analysis rather than treating improvement over the source-only model as equivalent to operational superiority over simple temporal forecasts.

For aggregation, seed-level forecasts were first summarized within each building. Buildings were then averaged with equal weight within their corresponding site, after which sites were given equal weight in the overall site-equal estimate. This prevented sites containing many more eligible buildings from dominating the primary summary solely because of cohort size.

---

## 2.7. Adaptation Outcome and Failure Definitions

In addition to average forecasting performance, adaptation-induced degradation, baseline underperformance, and seed-dependent directional instability were analyzed explicitly. The tolerance of \(10^{-12}\) was used solely to distinguish numerical ties and was not interpreted as a minimum practically meaningful degradation.

Four predefined failure categories were evaluated for each nonzero adaptation budget.

| Failure category | Definition | Interpretation |
|---|---|---|
| **F1** | Adapted MAE > source-only MAE | Adaptation-induced MAE degradation relative to source-only |
| **F2** | Adapted RMSE > source-only RMSE | Adaptation-induced RMSE degradation relative to source-only |
| **F3-PD** | Adapted MAE > previous-day MAE | Baseline underperformance relative to the previous-day baseline |
| **F3-W** | Adapted MAE > weekly-same-hour MAE | Baseline underperformance relative to the weekly baseline |
| **F3-any** | F3-PD or F3-W | Underperformance against at least one temporal baseline |
| **F3-both** | F3-PD and F3-W | Underperformance against both temporal baselines |
| **F4** | Different seeds for the same building and budget produced both MAE improvement and MAE degradation relative to source-only | Seed-dependent directional instability |

Because the source-only comparator already contains knowledge learned from the source buildings, F1 and F2 do not quantify negative transfer in the conventional target-only-versus-transfer sense. Instead, they quantify performance degradation associated with adding target adaptation to an already transferred source model.

F1 denotes adaptation-induced MAE degradation relative to the corresponding source-only model, whereas F2 denotes the analogous degradation in RMSE.

F1–F3 were evaluated at the building × adaptation-budget × seed level. F4 was evaluated at the building × adaptation-budget level and required all three predefined seeds to complete successfully. Cases in which MAE improved but RMSE worsened were recorded separately as metric-direction disagreements rather than being absorbed into one of the four primary failure categories.

Representative failure cases were selected using rules fixed during development rather than by visually choosing favorable examples after the final results were known. Repeated F1 cases, F4 cases, and MAE-improved/RMSE-worsened cases were therefore selected using predetermined ranking and tie-breaking rules.

**[Figure 2 about here: Failure taxonomy and success criteria]**

**Figure 2.** Conceptual distinction between source-only improvement, temporal-baseline superiority, adaptation-induced degradation, and seed-dependent instability.

---

## 2.8. Hypotheses and Statistical Analysis

The statistical analysis was based on hypotheses developed from the development cohort and frozen before the final evaluation electricity values were evaluated. These hypotheses should therefore be interpreted as development-informed hypotheses tested under a subsequently fixed final-evaluation protocol, rather than as hypotheses preregistered before all development analyses.

The primary hypothesis, **H1**, examined whether 28-day target adaptation reduced MAE relative to the source-only condition. For each building, the three seed-level differences in MAE were first averaged. These building-level values were averaged within each site, and the resulting site-level effects were given equal weight. H1 was considered supported only if the site-equal mean difference was below zero and the upper bound of its site-cluster bootstrap 95% confidence interval was also below zero. RMSE was reported as a secondary complementary metric and did not determine the H1 decision.

**H2** examined whether the site-equal F1 event rate decreased from the 1-day to the 28-day adaptation budget. The effect size was defined as the site-equal F1 event rate at 28 days minus the corresponding rate at 1 day. A negative value therefore indicated fewer F1 events at the larger adaptation budget.

**H3** examined the difference between two definitions of adaptation success at 28 days. Criterion A represented improvement over the source-only model, whereas criterion B required the adapted model to outperform both the previous-day and weekly-same-hour baselines. The H3 effect size was calculated as the site-equal rate of A minus the site-equal rate of B.

For uncertainty estimation, sites were treated as the resampling clusters. Percentile bootstrap confidence intervals were calculated from 10,000 site-level resamples using random seed 20260925, while retaining the buildings belonging to each sampled site. This approach preserved the nested site structure rather than treating the 672 hourly targets or the three random seeds as independent observations. The frozen protocol required at least five eligible final sites for cluster-based confirmatory inference.

H2 and H3 were reported as secondary effect-size analyses with bootstrap confidence intervals. No p-values or Holm-adjusted hypothesis tests were calculated for these two analyses in the final evaluation.

---

## 2.9. Exploratory Building-Characteristic Analysis

An exploratory analysis was conducted to investigate whether observable building characteristics were associated with adaptation outcomes. This analysis was separated from the confirmatory hypotheses and was not used to exclude buildings, modify adaptation budgets, or tune the forecasting model.

Only information available before the corresponding test interval was used. Static metadata included site identity and primary space use. Time-series characteristics included the mean, population standard deviation, coefficient of variation, and zero-value ratio calculated from available pre-test electricity observations. Daily repeatability was estimated using a 24-hour lag Pearson correlation when at least 24 valid lagged pairs were available, whereas weekly repeatability used a 168-hour lag with at least 168 valid pairs. A scale-difference feature was also calculated from the difference between the target-building mean available before testing and the fixed source normalization reference.

The characteristics were calculated separately for each adaptation budget according to the amount of information that would actually have been available at that point in deployment. The test-period target values were never included in characteristic calculation. Missing continuous characteristics were not imputed, and small categorical groups were not merged solely to obtain stronger associations.

Spearman rank correlations were calculated between continuous building characteristics and adaptation outcomes, including MAE and RMSE changes and failure-event rates. Site and building-use categories were summarized descriptively. Because multiple characteristics were explored and the analysis was not designed as a confirmatory causal analysis, no feature was interpreted as a causal determinant or definitive predictor of transfer success.

---

## 2.10. Edge Deployment Feasibility Validation

A separate engineering validation was conducted to determine whether the frozen forecasting architecture could be reproduced and executed on compact edge hardware. This experiment was not used to assess forecasting accuracy or to select a better forecasting model.

The deployment benchmark used the canonical source-only LSTM checkpoint corresponding to seed 20260924. The model contained 5,912 parameters and was executed in FP32 with batch size 1, input shape \((1,24,6)\), and output shape \((1,24)\). Head-only target adaptation does not change the LSTM architecture, tensor dimensions, or parameter count; therefore, this benchmark characterizes the computational requirements of the shared inference architecture rather than the forecasting accuracy of a particular adapted target checkpoint.

Before performance benchmarking, the checkpoint was transferred to an NVIDIA Jetson Orin Nano and compared with frozen RTX 3080 reference outputs. Sixty-four validation inputs were evaluated using CUDA FP32. Numerical consistency was accepted only when output shape matched exactly, no NaN or Inf values were present, normalized maximum and mean absolute differences remained below \(10^{-4}\) and \(2\times10^{-5}\), respectively, and physical-scale maximum and mean absolute differences remained below 0.05 and 0.01. These thresholds were fixed before observing the Jetson comparison results.

Latency was subsequently measured on the Jetson Orin Nano and an RTX 3080 using the same fixed benchmark vector. Two latency modes were evaluated. **Model-only latency** measured the forward pass with the input tensor already resident on the GPU. **Deployment end-to-end latency** included CPU-to-GPU transfer, the model forward pass, and GPU-to-CPU output transfer, while excluding disk I/O, Python startup, and checkpoint loading. Each mode used 200 warm-up iterations followed by five independent runs of 1000 measured inferences, producing 5000 observations per mode and device. A one-second stabilization interval was used between runs, and no outlier observations were removed.

CUDA memory statistics were reset before latency testing and peak allocated and reserved memory were recorded after the benchmark. Power and thermal measurements were performed separately from the latency experiment using three consecutive 60-second phases: idle, sustained batch-1 FP32 inference, and cooldown. Jetson telemetry was collected using `tegrastats`, including the reported VDD_IN input rail, whereas RTX power was obtained using `nvidia-smi` GPU board-power reporting. Because these two measurements represent different electrical boundaries, no cross-device power ratio or energy-per-inference comparison was calculated.

The same numerical-consistency check was repeated after performance benchmarking to verify that the benchmark had not altered model outputs. No model retraining, target adaptation, FP16 conversion, ONNX export, TensorRT optimization, quantization, pruning, or compilation optimization was performed during the edge validation.

**[Figure 3 about here: Edge deployment validation workflow]**

**Figure 3.** Edge deployment validation workflow comprising frozen-checkpoint transfer, cross-device FP32 numerical-consistency testing, latency and memory benchmarking, and separate power/thermal telemetry.

# 3. Results

## 3.1. Final Cohort and Integrity Checks

The predefined final evaluation proposal contained 510 buildings from nine sites. Application of the frozen quality-control criteria excluded 57 buildings, all because of continuous zero-valued periods of at least 30 days. The final evaluation cohort therefore comprised **453 buildings across seven sites**: Bobcat (22), Bull (111), Eagle (94), Gator (63), Hog (135), Moose (10), and Swan (18). No replacement buildings or alternative evaluation periods were introduced after cohort filtering.

All 453 retained buildings passed the structural integrity checks. The final forecasting execution produced **6,088,320 raw prediction rows without execution errors**, and independent metric recalculation agreed with the stored execution metrics to a maximum absolute difference of \(1.71\times10^{-13}\), below the predefined tolerance of \(10^{-10}\). No model settings, buildings, evaluation periods, seeds, learning rates, or epoch limits were changed after the final execution.

Additional leakage and consistency checks confirmed that all adaptation budgets and seeds for a given building used identical forecast origins, target timestamps, horizons, and observed target values. No overlap was detected between adaptation targets and test targets. The frozen source normalization statistics and checkpoint identities were also preserved, and all building-characteristic variables used only information available before the corresponding test period.

**[Table 2 about here: Final-test cohort by site]**

| Site | Retained buildings |
|---|---:|
| Bobcat | 22 |
| Bull | 111 |
| Eagle | 94 |
| Gator | 63 |
| Hog | 135 |
| Moose | 10 |
| Swan | 18 |
| **Total** | **453** |

---

## 3.2. Target-Adaptation Data-Efficiency Curve

Forecasting error decreased as the target-adaptation budget increased from 0 to 28 days. The site-equal MAE decreased from **40.6500** in the source-only condition to **38.4238, 34.9867, 29.6476, 27.4135, and 24.1671** after 1, 3, 7, 14, and 28 days of adaptation, respectively. The corresponding RMSE decreased from **54.6597** to **35.2364** at the 28-day budget.

Relative to the source-only condition, the site-equal MAE differences were −2.2262, −5.6633, −11.0024, −13.2365, and −16.4829 for 1, 3, 7, 14, and 28 days, respectively. At 28 days, this corresponded to an approximately **40.55% reduction in site-equal MAE** relative to the source-only model.

**[Figure 4 about here: Adaptation data-efficiency curve]**

**Figure 4.** Site-equal MAE and RMSE across target-adaptation budgets of 0, 1, 3, 7, 14, and 28 days.

**[Table 3 about here: Forecasting performance across adaptation budgets]**

| Adaptation budget | Site-equal MAE | Site-equal RMSE | ΔMAE vs. source | ΔRMSE vs. source | Source-improvement rate (A) | Beats-both-baselines rate (B) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 days | 40.6500 | 54.6597 | 0 | 0 | — | 0.0682 |
| 1 day | 38.4238 | 52.1976 | −2.2262 | −2.4621 | 0.3968 | 0.0267 |
| 3 days | 34.9867 | 48.2985 | −5.6633 | −6.3612 | 0.6923 | 0.0756 |
| 7 days | 29.6476 | 42.1623 | −11.0024 | −12.4975 | 0.8983 | 0.1587 |
| 14 days | 27.4135 | 39.3044 | −13.2365 | −15.3554 | 0.9192 | 0.1993 |
| 28 days | **24.1671** | **35.2364** | **−16.4829** | **−19.4233** | **0.9484** | **0.2685** |

Despite the reduction relative to the source-only model, the site-equal MAE of the 28-day adapted model remained higher than those of the two deterministic temporal baselines. The previous-day baseline achieved an MAE of **20.8740**, whereas the weekly-same-hour baseline achieved an MAE of **18.1359**. Their RMSE values were 39.5247 and 30.0582, respectively. Thus, at 28 days, the adapted model had a lower RMSE than the previous-day baseline but remained above the weekly-same-hour baseline in both MAE and RMSE.

---

## 3.3. Confirmatory and Secondary Hypothesis Results

For **H1**, all seven retained sites showed a negative 28-day MAE difference relative to the source-only condition. Site-level mean differences ranged from −2.1109 at Bobcat to −56.3351 at Moose. The site-equal 28-day effect was **−16.4829 MAE units**, with a 95% site-cluster bootstrap confidence interval of **[−30.9076, −6.5630]**. Because both the point estimate and the upper confidence bound were below zero, H1 satisfied the predefined support criterion.

For **H2**, the site-equal F1 rate decreased between the 1-day and 28-day adaptation budgets. The estimated difference, \(F1_{28d}-F1_{1d}\), was **−0.5515**, with a 95% bootstrap confidence interval of **[−0.6796, −0.4552]**. The direction was negative for each of the seven sites. No p-value or Holm-adjusted test was performed for this secondary analysis.

For **H3**, the site-equal rate for criterion A—improvement over the source-only model—was **0.9484** at 28 days, whereas criterion B—outperforming both deterministic baselines—was **0.2685**. The site-equal difference between these criteria was **0.6799**, with a 95% bootstrap confidence interval of **[0.5836, 0.7835]**.

At the building × seed level, 343 of 1359 cases both improved over source-only and outperformed both baselines. By contrast, **936 cases improved over the source-only model but did not outperform both baselines**. Sixty-seven cases satisfied neither criterion, while 13 cases did not improve over the source-only model but nevertheless outperformed both temporal baselines.

**[Figure 5 about here: H1–H3 summary]**

**Figure 5.** Summary of the primary and secondary final evaluation effects: (a) site-level 28-day MAE difference relative to source-only, (b) F1 rates at 1 and 28 days, and (c) source-improvement rate versus both-baselines success rate at 28 days.

---

## 3.4. Adaptation Failure Analysis

Adaptation-induced degradation relative to the source-only model decreased substantially as more target data became available. For MAE-based adaptation degradation (F1), 813 of 1359 cases (59.82%) exhibited higher error than the corresponding source-only condition at the 1-day budget, compared with 80 of 1359 cases (5.89%) at 28 days. The corresponding RMSE-based degradation counts (F2) decreased from 792 (58.28%) to 69 (5.08%).

Seed-dependent directional instability (F4) also declined with adaptation budget. F4 was observed in **164 of 453 building-budget cases (36.20%)** at 1 day and in **30 of 453 cases (6.62%)** at 28 days. Across all nonzero adaptation budgets, 448 of 2265 building-budget cases exhibited F4.

Underperformance relative to the deterministic temporal baselines remained more common than adaptation-induced degradation relative to the source-only model. At 28 days, 748 of 1359 building-seed cases failed to outperform the previous-day baseline and 691 failed to outperform the weekly-same-hour baseline. A total of **436 cases (32.08%)** remained worse than both baselines simultaneously, while 1003 cases were worse than at least one of the two baselines.

Across all nonzero budgets, the analysis recorded 1508 F1 events, 1399 F2 events, 4829 F3-PD events, 4505 F3-W events, and 3506 F3-both events. In addition, **128 cases showed an improvement in MAE accompanied by a worsening in RMSE**, demonstrating that the direction of adaptation effects was not always identical across the two error metrics.

**[Figure 6 about here: Adaptation-degradation, baseline-underperformance, and seed-instability rates across target-data budgets]**

**Figure 6.** Adaptation-degradation, baseline-underperformance, and seed-instability rates across target-data budgets.

**[Table 4 about here: Failure counts across adaptation budgets]**

| Budget | F1 | F2 | F3-PD | F3-W | F3-any | F3-both | F4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 day | 813 | 792 | 1211 | 1119 | 1316 | 1014 | 164 |
| 3 days | 377 | 350 | 1097 | 1000 | 1256 | 841 | 149 |
| 7 days | 134 | 101 | 921 | 892 | 1161 | 652 | 61 |
| 14 days | 104 | 87 | 852 | 803 | 1092 | 563 | 44 |
| 28 days | 80 | 69 | 748 | 691 | 1003 | 436 | 30 |

The predefined representative cases identified **Hog_office_Roger** as the repeated-F1 example, **Eagle_education_Brooke** at the 14-day budget as the largest seed-direction range under F4, and **Hog_office_Gustavo** at the 1-day budget as the selected MAE-improved/RMSE-worsened case. These examples were selected using rules fixed before the final evaluation results rather than by post hoc visual selection.

---

## 3.5. Exploratory Building-Characteristic Analysis

The exploratory characteristic dataset contained **2718 rows**, corresponding to 453 buildings evaluated at six adaptation budgets. Of these, 2265 rows contained adaptation-based longer-history characteristics, and weekly-repeatability measures were available for 905 rows. Thirteen recorded primary-space-use categories were present, while 18 Swan buildings had missing official space-use metadata and were retained as an explicit `<missing>` category.

The associations varied by adaptation budget rather than identifying one characteristic that dominated at every budget. At the 1-day budget, daily repeatability showed the largest reported absolute Spearman association with \(\Delta\mathrm{MAE}_{source}\) (\(\rho=-0.4063\), \(n=389\)). At 3 days, the largest association was observed for the absolute difference between the available target mean and the source scale reference (\(\rho=-0.4128\), \(n=453\)). At 7 days, the coefficient of variation of the available target history had \(\rho=0.3185\).

At the larger budgets, load-scale characteristics showed moderate negative associations with \(\Delta\mathrm{MAE}_{source}\). At 14 days, the 24-hour context mean had \(\rho=-0.3225\), while at 28 days the percentile of the target mean among the 20 source-building means, the available mean, and the initial 24-hour context mean each had correlations of approximately −0.35. These analyses were exploratory; no confirmatory multiple-testing procedure was applied and no characteristic was used as a building-selection or adaptation rule.

A post-analysis software review identified that the initial exploratory grouping operation had omitted 18 Swan buildings with missing `primaryspaceusage` values because of the default missing-value handling of the grouping implementation. The exploratory characteristic tables were regenerated with these records preserved as `<missing>`. This correction affected only the exploratory characteristic summaries and did not alter the cohort, predictions, MAE/RMSE values, F1–F4 results, or H1–H3 outcomes.

**[Figure 7 about here: Exploratory characteristic associations]**

**Figure 7.** Selected exploratory Spearman associations between pre-test building characteristics and adaptation-induced MAE changes across target-data budgets.

---

## 3.6. Edge Deployment Results

The frozen canonical source-only LSTM was successfully reproduced on the Jetson Orin Nano using CUDA FP32. Across 64 validation vectors comprising 1536 output values, the maximum normalized absolute difference from the frozen RTX reference was **\(1.4782\times10^{-5}\)** and the mean absolute difference was **\(1.4947\times10^{-6}\)**. On the physical scale, the maximum and mean absolute differences were **0.00524** and **0.000530**, respectively. No NaN or Inf values were observed, and all predefined numerical-consistency thresholds were satisfied.

Under the frozen batch-1 FP32 latency benchmark, the Jetson Orin Nano achieved a model-only median latency of **0.6756 ms** and P95 latency of **0.7311 ms**. Deployment end-to-end latency had a median of **0.8831 ms** and a P95 of **0.9373 ms**. The corresponding RTX 3080 medians were 0.1711 ms for model-only inference and 0.2326 ms for end-to-end inference. Peak CUDA allocated and reserved memory were **17.24 MiB** and **22.00 MiB**, respectively, on both devices.

One Jetson end-to-end observation of **195.8847 ms** occurred during the fifth run. In accordance with the frozen benchmark protocol, this observation was retained and no benchmark repetition or outlier deletion was performed. The Jetson end-to-end mean was therefore 0.9330 ms, whereas the median and P95 remained below 1 ms.

During the separate sustained-workload telemetry experiment, the Jetson-reported VDD_IN power increased from **3.146 W** at idle to **4.031 W** during sustained inference, with a reported peak of 4.032 W. The sustained mean and peak GPU temperatures were **50.329 °C** and **50.843 °C**, respectively. For the RTX 3080, `nvidia-smi` reported 63.373 W at idle, 70.422 W during the sustained workload, and a peak board-power value of 123.660 W. Because Jetson VDD_IN and RTX GPU board power represent different measurement boundaries, no cross-device power-efficiency ratio was calculated.

Pre- and post-benchmark numerical-consistency checks produced identical acceptance outcomes, with zero NaN and Inf values on both platforms. RTX process RSS was unavailable because of a Windows collection-code error; the latency, CUDA-memory, telemetry, and numerical-consistency results were unaffected, and the benchmark was not rerun to replace the missing measurement.

**[Table 5 about here: Edge deployment benchmark]**

| Metric | Jetson Orin Nano | RTX 3080 |
|---|---:|---:|
| Model-only mean latency (ms) | 0.6906 | 0.2052 |
| Model-only median (ms) | 0.6756 | 0.1711 |
| Model-only P95 (ms) | 0.7311 | 0.3544 |
| End-to-end mean latency (ms) | 0.9330 | 0.3374 |
| End-to-end median (ms) | 0.8831 | 0.2326 |
| End-to-end P95 (ms) | 0.9373 | 0.9128 |
| Peak CUDA allocated (MiB) | 17.24 | 17.24 |
| Peak CUDA reserved (MiB) | 22.00 | 22.00 |
| Process RSS | ~559 MiB | Unavailable |
| Reported sustained power | 4.031 W, VDD_IN | 70.422 W, GPU board power |
| Peak GPU temperature | 50.843 °C | 63.000 °C |

**[Figure 8 about here: Edge latency distribution]**

**Figure 8.** Distribution of model-only and deployment end-to-end FP32 inference latency on the Jetson Orin Nano and RTX 3080 under the frozen batch-1 benchmark protocol.

# 4. Discussion

## 4.1. Target Adaptation Benefits and Data Efficiency

The final evaluation showed a clear aggregate relationship between the amount of target-building data and cross-building forecasting performance. As the adaptation budget increased from 0 to 28 days, the site-equal MAE decreased from 40.65 to 24.17, corresponding to an approximately 40.5% reduction relative to the source-only model. RMSE followed the same overall direction. Notably, all seven final sites exhibited a negative 28-day MAE difference relative to the source-only condition, and the site-cluster confidence interval for the overall effect remained below zero.

This result is consistent with the broader cross-building transfer-learning literature showing that target-domain observations can improve a model initially trained on other buildings [2–8]. Earlier work has examined transfer under limited target data, few-shot adaptation, and the influence of target-data availability [4–8]. The present results extend this line of evidence by evaluating a fixed sequence of 0-, 1-, 3-, 7-, 14-, and 28-day budgets on the same target test periods rather than comparing only a small number of zero-shot and fine-tuned conditions. This design makes the amount of target data itself an explicit experimental dimension.

The results also indicate that the value of adaptation cannot be summarized by the largest budget alone. Although the aggregate site-equal error decreased at every evaluated budget, adaptation reliability differed markedly across the data range. At the 1-day budget, the site-equal MAE was already lower than the source-only MAE, yet **813 of 1359 building-seed cases exhibited higher MAE than their corresponding source-only runs**. At 28 days, the number of such F1 events decreased to 80 of 1359.

This apparent contrast is important. An improvement in an aggregate mean does not imply that the majority of individual target adaptations are beneficial. A comparatively small number of large improvements can coexist with many smaller degradations, and site-equal aggregation introduces an additional level at which effects are summarized. Consequently, statements such as “one day of adaptation improves transfer performance” would be too broad for these results. A more accurate interpretation is that even very limited target information can improve the aggregate error under the present protocol, but **the reliability of adaptation improves substantially as more target data become available**.

The 1-day condition is particularly informative because it contained only one complete supervised adaptation window, whereas 3, 7, 14, and 28 days provided 49, 145, 313, and 649 rolling windows, respectively. The sharp reduction in adaptation-degradation events as the adaptation set expanded suggests that the amount of target supervision is relevant not only to average accuracy but also to the stability of the transfer process. This complements prior few-shot studies [6–8] by emphasizing failure-event frequency as an additional dimension of data efficiency rather than considering prediction error alone.

---

## 4.2. Improvement over Source-Only Transfer Does Not Imply Baseline Superiority

The most consequential distinction in the present results is between **relative transfer success** and **absolute forecasting competitiveness**.

At the 28-day budget, the site-equal rate of improving over the source-only model was approximately **94.8%**, whereas the rate of outperforming both the previous-day and weekly-same-hour baselines was only **26.8%**. At the building-seed level, 936 of 1359 cases improved over the source-only model but nevertheless failed to outperform both temporal baselines.

The aggregate errors show the same distinction. The 28-day adapted LSTM achieved an MAE of 24.17, compared with 40.65 in the source-only condition, but the previous-day and weekly-same-hour baselines achieved MAEs of 20.87 and 18.14, respectively. In RMSE, the adapted model at 35.24 performed better than the previous-day baseline at 39.52 but remained above the weekly baseline at 30.06.

These results demonstrate why transfer-learning studies benefit from retaining strong simple forecasting references. An adaptation method can be highly effective according to an internal transfer criterion while still not providing the lowest forecasting error available from an inexpensive deterministic rule. General forecasting practice likewise emphasizes the use of meaningful naïve or reference forecasts when assessing whether a more complex model provides practical predictive value [11].

This does not mean that the temporal baselines and the LSTM constitute perfectly matched information sets. In particular, the weekly-same-hour baseline directly accesses the observation from 168 h before the target time, whereas the LSTM input was restricted to the most recent 24 h of electricity history plus deterministic calendar variables. The weekly baseline is therefore best interpreted as a **strong operational comparator**, not as an equal-capacity or equal-input neural-network baseline.

This distinction is also useful for interpreting why the weekly baseline remained competitive. Electricity use in many buildings may exhibit recurrent weekly schedules that can be exploited directly by a seven-day persistence rule. Under the present input design, the LSTM had to infer transferable temporal structure without direct access to a 168-h historical context. Future work could evaluate whether explicitly supplying longer historical context or weekly-lag features reduces this gap, but that question was not tested in the present study.

The current results therefore support a conservative operational interpretation: **target adaptation substantially improves a transferred model, but the improvement should not automatically be interpreted as evidence that transfer learning is preferable to all simpler forecasts for a given building.**

---

## 4.3. Adaptation-Induced Degradation, Seed Instability, and Heterogeneous Outcomes

The explicit failure analysis further demonstrates that cross-building adaptation is heterogeneous. Across all nonzero budgets, 1508 MAE-based adaptation-degradation events (F1) and 1399 RMSE-based adaptation-degradation events (F2) were observed. Seed-dependent directional instability occurred in 448 of 2265 building-budget combinations. In addition, 128 cases improved in MAE while worsening in RMSE.

These F1 and F2 events correspond to performance degradation relative to the source-only transferred model and are therefore narrower than definitions of negative transfer based on comparisons with independently trained target-only models.

The reduction in F1 from 59.8% at 1 day to 5.9% at 28 days, and the reduction in F4 from 36.2% to 6.6%, indicate that additional target data reduced the frequency of both adaptation-induced degradation and seed-dependent directional instability. However, neither phenomenon disappeared completely. Even at 28 days, some buildings and seeds responded unfavorably to adaptation.

These observations are consistent with the growing literature emphasizing transferability, source-target mismatch, robustness, and negative transfer rather than assuming uniformly beneficial adaptation [9,10]. The present study adds a complementary perspective by observing these failures across predefined adaptation budgets and repeated seeds. Rather than reporting only the mean performance of an adapted model, F1–F4 preserve several distinct ways in which adaptation can fail: worsening relative to source-only performance, failure to outperform simple temporal forecasts, and changes in the direction of adaptation across random seeds.

The exploratory characteristic analysis did not identify a single building characteristic that consistently explained adaptation behavior across every data budget. For example, daily repeatability showed a Spearman correlation of −0.406 at 1 day, whereas measures related to load magnitude or source-scale difference were among the larger associations at other budgets. At 28 days, several load-scale measures showed correlations close to −0.35 with the change in MAE.

These findings should not be interpreted as evidence that a particular building characteristic causally determines transfer success. First, the characteristic analysis was exploratory and involved multiple candidate variables. Second, the same building was represented across multiple adaptation budgets. Third, only seven independent final sites were available. Finally, a negative association between a scale-difference measure and \(\Delta\mathrm{MAE}_{source}\) does not necessarily imply that large source-target differences are inherently advantageous; buildings with larger initial source-model errors may simply have greater room for improvement after adaptation.

Accordingly, the exploratory results are more useful as **hypothesis-generating evidence** than as a building-selection rule. A future source-selection study could test prospectively whether temporal repeatability, load scale, operating schedule, or other pre-deployment descriptors predict adaptation benefit. This would be particularly relevant to similarity-guided transfer approaches such as those investigated in recent building-energy forecasting research [9].

---

## 4.4. Implications for Cross-Building Transfer-Learning Research

The results also clarify the relationship of this study to recent cross-building forecasting research. Transfer learning itself, target fine-tuning, and data-scarcity adaptation are already established topics in building energy forecasting [2–10]. In particular, Spencer et al. [5] evaluated multi-source transfer and target fine-tuning using Transformer-based architectures on BDG2, while recent studies have investigated few-shot prediction, target-data availability, domain adaptation, and transfer robustness [6–10].

The contribution of the present study is therefore not a new claim that transfer learning can be applied to buildings. Instead, the study isolates several evaluation questions that can otherwise be hidden by a single average accuracy value: how adaptation performance changes across a fixed target-data budget sequence; how often adaptation worsens the original source model; whether those effects are stable across repeated training; and whether internal transfer gains actually exceed strong temporal baselines. This scope was deliberately separated from claims of architectural novelty.

The use of a compact LSTM also serves this purpose. More complex models may achieve different absolute errors, and the present results do not imply that an LSTM is preferable to Transformer-based or other forecasting architectures. Rather, restricting the confirmatory experiment to one compact architecture reduces the number of changing factors and enables a clearer examination of the adaptation process itself.

This is particularly relevant when comparing absolute errors across studies. Existing studies differ in target buildings, temporal splits, input variables, normalization procedures, model architectures, and forecasting windows. Consequently, direct claims that the present MAE is better or worse than the MAE reported by another study would not be methodologically justified. The meaningful comparison is instead at the level of **experimental behavior and study design**: prior research establishes that transfer and fine-tuning can be useful, while the present study quantifies how their benefits and failure conditions change under one controlled fixed-test protocol.

---

## 4.5. Edge Deployment Feasibility and Practical Implications

The edge experiment provides an additional engineering perspective on the forecasting architecture. The frozen source-only LSTM reproduced the reference outputs on the Jetson Orin Nano within all predefined FP32 numerical-consistency limits. Typical inference latency was small relative to the forecasting task: the Jetson end-to-end median was approximately **0.883 ms** and the P95 was approximately **0.937 ms** under batch-1 FP32 execution. Peak CUDA allocated memory was approximately **17.24 MiB**.

For a forecasting application that generates the next 24 hourly values from one model invocation, this computational latency is negligible relative to the forecasting interval itself. The result therefore indicates that the evaluated compact architecture can be executed on the tested Jetson platform without requiring FP16, TensorRT, pruning, or quantization.

This conclusion is intentionally narrower than a claim of universal real-time suitability. The benchmark measured one compact model, one fixed benchmark vector, batch size 1, and FP32 execution. It did not include sensor acquisition, database access, networking, application logic, or a complete building-management pipeline. One 195.9 ms Jetson end-to-end observation was also retained according to the frozen protocol, indicating that occasional latency jitter was present even though the median and P95 remained below 1 ms.

The deployment result complements recent work combining building-energy forecasting and edge computation [12]. However, the present edge experiment should be interpreted as a **feasibility validation of the shared inference architecture**, not as an additional transfer-learning accuracy experiment. The deployed checkpoint was the canonical source-only model. Because target adaptation updated only the output head without changing architecture, tensor dimensions, or parameter count, the benchmark is informative about the computational requirements of the adapted architecture class, but it does not demonstrate the forecasting accuracy of every target-adapted checkpoint.

Power measurements require similar caution. The Jetson reported approximately 4.03 W during sustained inference through the VDD_IN measurement, whereas the RTX measurement was obtained from GPU board-power reporting. These quantities do not share the same electrical boundary, and therefore no cross-device power ratio or “times more efficient” claim is supported.

The practical implication is consequently modest but useful: **the forecasting architecture itself is sufficiently lightweight to be deployed on the tested edge hardware under the evaluated configuration**, leaving data availability and forecasting quality—not inference latency—as the more substantial challenges examined in this study.

---

## 4.6. Limitations and Future Work

Several limitations define the scope of these findings.

First, the final evaluation contained 453 buildings but only **seven independent sites**. The site-cluster bootstrap therefore satisfies the predefined minimum of five clusters but remains based on a small number of independent site-level units. Confidence intervals should consequently be interpreted with appropriate caution.

Second, the final cohort represents buildings that passed the predefined quality criteria. Fifty-seven proposed final evaluation buildings were excluded because of continuous zero-valued periods of at least 30 days, and two proposed sites retained no eligible buildings. The conclusions therefore apply to the retained quality-controlled cohort rather than to all BDG2 electricity records.

Third, the physical metering boundaries of all BDG2 electricity series could not be independently shown to be identical. Absolute electricity magnitudes and raw-unit MAE values may therefore reflect differences in building scale and meter scope as well as forecasting difficulty. This study mitigated sample-count imbalance through building- and site-level aggregation, but this does not eliminate physical heterogeneity among meters.

Fourth, the study evaluated **one compact LSTM architecture and one output-head-only adaptation strategy**. The findings regarding the adaptation curve and failure rates should not be assumed to hold quantitatively for full-network fine-tuning, alternative recurrent models, Transformers, domain-adversarial methods, or other transfer strategies. Future work should apply the same fixed-budget and failure-analysis protocol to multiple architectures rather than simply comparing their best average errors.

Fifth, the forecasting inputs were limited to the previous 24 h of electricity observations and deterministic calendar information. The potential contribution of weather, occupancy, indoor environmental variables, and longer historical contexts was not evaluated. Similarly, the weekly baseline used an observation from seven days earlier, whereas the neural model did not receive a 168-h historical context. Future work could therefore evaluate longer input windows or explicit daily and weekly lag features under the same leakage-controlled protocol.

Sixth, the final evaluation sites were separated from the development sites and their raw final evaluation electricity values were not used for model development before the fixed final run. However, the broader site-split construction had access to candidate-level quality information across the dataset. The final evaluation should therefore be described as a **site-disjoint fixed final evaluation**, not as a fully blind external validation.

Seventh, the building-characteristic analysis was exploratory. The observed correlations were neither preregistered causal hypotheses nor used to construct a failure predictor. Future work should test prospective source-selection or adaptation-risk models on additional sites using only information genuinely available before deployment.

Finally, the edge validation was restricted to a single canonical source-only LSTM checkpoint, batch-1 FP32 inference, and the tested Jetson Orin Nano and RTX 3080 environments. The Jetson `jetson_clocks` state could not be directly verified, the RTX environment operated under Windows WDDM with background graphical contexts, and RTX process RSS was unavailable because of a collection-code error. The two platforms also used different power-measurement boundaries. Future hardware work could evaluate multiple edge devices, precision modes, workloads, and standardized external power instrumentation; however, such optimization is separate from the forecasting questions examined here.

Taken together, these limitations constrain the generality of the absolute performance values but do not alter the central observation of the experiment: **under the fixed evaluation protocol, increasing target adaptation data substantially reduced error relative to the transferred source model and reduced the frequency of adaptation-induced degradation, while strong temporal baselines remained difficult to outperform consistently.**

# 5. Conclusions

This study investigated data-efficient cross-building transfer learning for direct 24-hour electricity forecasting, with particular emphasis on the amount of target-building data required for adaptation and the conditions under which adaptation can fail. A compact LSTM pretrained on source buildings was evaluated without target adaptation and after head-only adaptation using 1, 3, 7, 14, and 28 consecutive days of target-building observations. The final evaluation comprised 453 quality-controlled buildings across seven site-disjoint final evaluation sites under a fixed test protocol.

Increasing the target-data budget substantially reduced forecasting error relative to the source-only model. The site-equal MAE decreased from 40.65 without target adaptation to 24.17 after 28 days of adaptation, corresponding to an approximately 40.5% reduction. At the same time, the frequency of MAE-based adaptation degradation relative to the source-only model decreased from 59.8% of building-seed cases at the 1-day budget to 5.9% at 28 days, and seed-dependent directional instability also became less frequent as more target data became available. These findings indicate that target-data availability affects not only average forecasting accuracy but also the reliability of cross-building adaptation.

However, improvement over the transferred source model did not necessarily imply superiority over simple temporal forecasts. At 28 days, the site-equal source-improvement rate was approximately 94.8%, whereas the corresponding rate for outperforming both the previous-day and weekly-same-hour baselines was approximately 26.8%. The 28-day adapted model also remained above both temporal baselines in site-equal MAE. Therefore, cross-building transfer performance should be assessed using at least two distinct criteria: whether adaptation improves the transferred model itself and whether the resulting forecast is competitive with strong operational baselines.

The auxiliary hardware experiment further showed that the frozen compact inference architecture could reproduce the reference FP32 predictions on an NVIDIA Jetson Orin Nano within the predefined numerical-consistency tolerances. Under the tested batch-1 configuration, the Jetson achieved a median end-to-end inference latency of approximately 0.883 ms and a P95 latency of approximately 0.937 ms. These measurements support the computational feasibility of the evaluated architecture on the tested edge platform, but they do not imply forecasting-accuracy gains, universal real-time performance, or cross-device power-efficiency superiority.

Overall, the results show that **increasing target adaptation data can substantially reduce source-only cross-building forecasting error and lower the risk of adaptation-induced performance degradation, while relative transfer improvement alone is insufficient evidence of absolute forecasting superiority**. Future research should test whether these findings persist across additional independent sites, forecasting architectures, adaptation strategies, longer historical input contexts, and additional pre-deployment covariates. The same fixed-budget and failure-aware evaluation framework may also provide a useful basis for determining when target adaptation is practically justified and when a simpler temporal forecast remains the more competitive alternative.

# References

**[1]** Han, G.; Lee, D.; Kwon, M.; Park, J.; Kim, J.; Choi, J.; Bae, S.; Ko, J.; Park, S. *Research on AIoT-Based Intelligent Energy Digital Twin Platform for Zero-Energy Building*. **Journal of Internet of Things and Convergence** 2026, 12, 7–22. https://doi.org/10.20465/KIOTS.2026.12.2.007.  

**[2]** Ribeiro, M.; Grolinger, K.; ElYamany, H.F.; Higashino, W.A.; Capretz, M.A.M. Transfer learning with seasonal and trend adjustment for cross-building energy forecasting. *Energy and Buildings* **2018**, *165*, 352–363. https://doi.org/10.1016/j.enbuild.2018.01.034.

**[3]** Gao, Y.; Ruan, Y.; Fang, C.; Yin, S. Deep learning and transfer learning models of energy consumption forecasting for a building with poor information data. *Energy and Buildings* **2020**, *223*, 110156. https://doi.org/10.1016/j.enbuild.2020.110156.

**[4]** Fan, C.; Sun, Y.; Xiao, F.; Ma, J.; Lee, D.; Wang, J.; Tseng, Y.-C. Statistical investigations of transfer learning-based methodology for short-term building energy predictions. *Applied Energy* **2020**, *262*, 114499. https://doi.org/10.1016/j.apenergy.2020.114499.

**[5]** Spencer, R.; Ranathunga, S.; Boulic, M.; van Heerden, A.H.; Susnjak, T. Transfer learning on transformers for building energy consumption forecasting—A comparative study. *Energy and Buildings* **2025**, *336*, 115632. https://doi.org/10.1016/j.enbuild.2025.115632.

**[6]** Liu, Q.; Yu, D.; Xue, Y.; Sun, H.; Yang, S.; Han, F. A cross-building few-shot energy load forecasting framework via transferable temporal learning and residual correction. *Energy and Buildings* **2026**, *363*, 117530. https://doi.org/10.1016/j.enbuild.2026.117530.

**[7]** Wu, C.; Zhang, X.; Li, Y.; Chen, Z.; Zhang, Z. Cross-building energy consumption forecasting with deep transfer learning. *Journal of Building Engineering* **2026**, *117*, 114695. https://doi.org/10.1016/j.jobe.2025.114695.

**[8]** Lei, M.; Wang, Y.; Zhan, Y. TSDAM: A time-series domain adaptation model for building energy prediction under data scarcity. *Energy and Buildings* **2026**, *355*, 117017. https://doi.org/10.1016/j.enbuild.2026.117017.

**[9]** Choi, W.; Lee, S.; Langtry, M.; Choudhary, R. Privacy-preserving transfer learning framework for building energy forecasting with fully anonymized data. *Applied Energy* **2026**, *412*, 127600. https://doi.org/10.1016/j.apenergy.2026.127600.

**[10]** Bian, J. et al. On the robustness of short-term building energy forecasting under cross-climate transfer. *Energy and AI* **2026**, *25*, 100818. https://doi.org/10.1016/j.egyai.2026.100818.

**[11]** Petropoulos, F. et al. Forecasting: Theory and practice. *International Journal of Forecasting* **2022**, *38*, 705–871. https://doi.org/10.1016/j.ijforecast.2021.11.001.

**[12]** Iqbal, M.S.; Adnan, M. Edge computing and transfer learning-based short-term load forecasting for residential and commercial buildings. *Energy and Buildings* **2025**, *329*, 115273. https://doi.org/10.1016/j.enbuild.2025.115273.

**[13]** Miller, C.; Kathirgamanathan, A.; Picchetti, B.; et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data* **2020**, *7*, 368. https://doi.org/10.1038/s41597-020-00712-x.
