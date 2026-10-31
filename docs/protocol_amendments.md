# Protocol amendments (chronological)

1. 2026-10-10, before any formal Chorus result was inspected: engine optimizations (prioritized
   working sets, active-set cycling, Gram capacity doubling, masking converged responses, vectorized
   block norms). They change execution only; solutions are certified by the same duality gap.
   Invariance to be confirmed on tuning seeds (identical selections up to the certificate tolerance).
2. 2026-10-10: the one-standard-error selection is applied per response instead of once for all
   responses. Motivation: on a 40-response DepMap design draft, validation errors and model sizes
   (not test metrics) showed that a single global choice kept dense fits for every response.
   Decision evidence: tuning seeds 101 to 103 only (F1 0.961 vs 0.937, MCC 0.963 vs 0.940,
   R2 and FEE unchanged, AUPRC 0.992 vs 0.994). All formal Chorus results are recomputed with the
   final frozen code (configs/chorus_frozen.json, source hash recorded there).
3. Formal simulations: seeds 1001 to 1005 are completed first for every setting (seed-major order);
   seeds 1006 to 1010 are added if time allows, always for every setting at once.

Invariance check for amendment 1 (tuning seeds, global rule, new engine vs old engine):
F1 0.9365 vs 0.9368, MCC 0.9401 vs 0.9403, FEE 0.0671 vs 0.0665, R2 0.6126 vs 0.6129,
AUPRC 0.9940 vs 0.9943, identical distribution of selected configurations. Differences stem from
solutions that agree within the certificate tolerance but differ slightly in validation-selected
levels for a few responses.
4. 2026-10-10: timing instrumentation (time spent in shared data passes and Gram construction is
   recorded). No change to computations; hash updated in configs/chorus_frozen.json.
5. Methodological presentation: the batching-exactness Proposition and its proof sketch were removed
   from the manuscript; exactness is stated as an invariant in the method section (PVLDB systems
   papers state correctness in prose and defer proofs to technical reports).
6. 2026-10-10: the granularity of the one-standard-error rule (one choice for all responses or one per
   response) is decided automatically from the validation split (20 random halves; the variant with
   smaller models is adopted unless its held-out error exceeds the other's by more than one standard
   error). Motivation, disclosed: after amendment 2, the DepMap comparison case showed lower test R2
   for Chorus with per-response selection (0.086) than for the group lasso solvers (0.111), and a
   validation-only analysis (held-out halves of the validation split, no test data) showed that
   per-response choices overfit when the validation split is small relative to the number of
   candidates (DepMap: per-response 0.917 vs shared 0.885 normalized held-out error; Beijing: 0.411
   vs 0.418; tuning simulations: 0.398 vs 0.395). A first variant that compared held-out error alone
   was stopped after 7 tuning cases (it chose the shared rule in all of them); the adopted variant
   applies the same parsimony principle as the candidate rule. Decision evidence, tuning seeds only
   (36 cases): R2 0.6131, FEE 0.0661, F1 0.947, MCC 0.950, AUPRC 0.9935 (shared on 30 cases,
   per-response on 6), against per-response F1 0.961 / FEE 0.0666 and shared F1 0.937 / FEE 0.0671.
   All formal and real-data Chorus results are recomputed with this code; the earlier per-response
   outputs are kept as Chorus_resp.
7. 2026-10-10: the Beijing case is regenerated with the released basis code. The original case
   was written about 20 seconds before the uniform-knot fallback for tied predictors was added to
   src/chorus/basis.py, so its five rainfall terms had 3 instead of 5 columns and the released code
   would not reproduce it. Raw predictors, responses and splits are bit-identical (verified); every
   method is rerun on the regenerated case (SAM reads the raw predictors and is unaffected in
   principle). The earlier case and outputs are kept as results/cases/beijing_v0 and
   results/real/beijing_v0. The DepMap case reproduces bit for bit with the released code.
8. 2026-10-10: basis-size diagnostic (planned sensitivity to d): Chorus and skglm are also run on
   d=3 copies of both real cases (the basis size of SAM) to explain the accuracy gap to SAM. The
   main comparison keeps the frozen design (d=5); the diagnostic is reported as sensitivity.
9. 2026-10-10: robustness experiment added at the authors' request after the formal results were
   known (design fixed before any run in planning/robustness_design.md; seeds 3001 to 3005; frozen
   code, no tuning; all situations and metrics reported).
10. 2026-10-10: timing harness fixes. (a) The disk cases lacked the raw training predictors that
   SAM needs, so SAM failed on every row-scaling size; the predictors are regenerated exactly from
   the per-chunk seeds (checked against the stored validation predictors) and SAM is rerun.
   (b) The first repetition of Chorus at q = 256 overlapped figure generation on the same machine
   (pass time 3.6 times that at q = 64 per pass); every method receives a second repetition at
   q = 256 and the minimum over repetitions is reported, as for all other items. From this point,
   no multi-threaded work runs on the machine during timing.
11. 2026-10-10: the disk-resident design reader used a memory map; on Windows the touched pages
   stay in the working set that the timing harness measures, so resident memory grew with the
   design (Chorus stopped at the 2 GB budget for 100k rows). The reader now uses explicit
   unbuffered file reads into chunk-sized buffers. Reads are bit-identical (checked on the 25k-row
   case; X^T R agrees to 3e-13) and all unit tests pass; a full streamed pass over the 1.86 GB
   design peaks at 346 MB. Execution only, estimator unchanged; hash updated. All Chorus
   disk-based timing runs (row scaling, batch size, memory budget) are rerun with the new reader;
   the earlier runs and log are kept (Chorus_mmap, timing_log_before_reader_fix.jsonl).
12. 2026-10-11: the timing runner skipped every larger setting after a stop in the same experiment.
   This is right for response and row scaling but wrong for the memory budget (a larger budget is
   easier): after Chorus reached the time cap at 0.25 GB, the budgets 0.5 to 4 GB were skipped.
   The rule now applies only to response and row scaling; the skipped budget settings are run
   (the 0.25 GB stop is kept and reported).
13. 2026-10-11: settings skipped after a stop are run. Under the skip rule, a baseline stopped at
   one size of response or row scaling was not run at larger sizes, which left those cells without
   a measurement. At the authors' request that every table cell report a measurement, each such
   setting is run once with the same harness, budget and cap (run_timing.py --no-skip,
   configs/timing_plan_stops.json), and the table reports the measured time and resident memory at
   the stop. Methods, budgets and caps are unchanged; no Chorus run is involved.
14. 2026-10-12, major revision (reviewer comments 3 and 4, author decision of 12 Oct), written before any
   new Chorus result was computed. (a) Weights use scale-free norms a_jr = ||beta_jr|| / sd_train(y_r).
   (b) The selection rules of amendments 2 and 6 (per-response 1-SE, and the 20-halves choice between
   shared and per-response selection) are replaced by cross-fitted empirical-Bayes selection
   (src/chorus/selection.py): validation rows are split into two folds (seeded random halves of rows for
   exchangeable data; alternating calendar weeks for Beijing, with weeks as clusters); stage 1 and stage 2
   track the level selected on V1, on V2 and on all of V; every candidate is scored, cluster by cluster,
   by the fit selected on the other fold; paired differences to a reference candidate get cluster-robust
   variances; a DerSimonian-Laird random-effects model across responses shrinks each response's
   difference; each response adopts the sparsest candidate within kappa posterior standard errors of its
   best one; the final fit of the chosen candidate uses weights and level selected on all of V.
   (c) Candidates: GL and the ten (alpha, gamma) configurations for each basis size d in {5, 3}
   (22 candidates); every comparison method also chooses d in {3, 5} by validation (biglasso jointly
   over responses, SAM with its own basis of 3 or 5 functions).
   (d) Engine change without effect on the estimator: solve rounds skip columns that have no KKT
   violation and a closed gap; improvement thresholds of the selected level are relative (1e-10).
   Tuning, pre-registered here: on tuning seeds 101 to 103 only (12 settings x 3 seeds), one Chorus run
   per case saves all candidates; the selection rule is then varied offline over kappa in {0.5, 1},
   parsimony by groups or by parameters, and reference GL(d=5) or the better GL per response
   (8 variants). Objective: mean over cases of the average rank of F1, MCC, FEE and test R2 among the
   8 variants; ties go to kappa = 1, groups, GL(d=5). No-shrinkage and full-pooling versions of the
   chosen rule are reported as anchors and are not candidates. The Beijing fold and cluster scheme is
   fixed by design and not tuned. Environment: adelie 1.1.52 installed (int64 dtype patch for numpy 1.x
   on Windows, no numerical change); scipy updated to 1.17.1 as an adelie dependency (existing cases
   are not regenerated).
15. 2026-10-12, tuning result for amendment 14 (tuning seeds only, results/pilot/tune_selection.csv):
   mean ranks 3.74 (kappa 0.5, groups, GL d=5 reference), 3.78, 3.82, 3.85 (other kappa 0.5 variants),
   4.88 to 5.51 (kappa 1). Frozen: kappa = 0.5, parsimony by groups, reference GL(d=5). Tuning-seed means
   of the frozen rule: F1 0.974, MCC 0.975, FEE 0.0633, R2 0.6138 (previous frozen Chorus 0.919, 0.924,
   0.0668, 0.6128). Selected candidates: GL 0.2%, alpha 0 18%, intermediate alpha 60%, alpha 1 22%;
   d = 3 for 67% of responses.
16. 2026-10-12, engine changes for reviewer comments 1, 5 and 7 (execution only): (a) memory ledger
   (src/chorus/ledger.py): M bounds the peak working set of the fitting process; M0 is measured at the start
   of the fit; batch arrays, full-q arrays, read buffers and the Gram matrix (with growth transient and a
   per-response copy) are charged with a 10% safety factor and a margin delta = max(64 MB, 0.05 M); the
   union working set has a width cap; on overflow the engine evicts unneeded blocks, then splits the batch
   along the leading principal direction of its candidate pattern and restarts both halves (cap recomputed
   for the smaller batch), and reports an infeasible budget for a single response; stage 2 falls back to the
   streaming engine when its Gram matrix does not fit; option stage2_shared=False forces that path (used for
   the sequential and unshared executions of the same estimator). (b) Disk reads bypass the OS cache
   (FILE_FLAG_NO_BUFFERING, src/chorus/winio.py); requested, issued and device bytes are recorded; full scans
   prefetch the next column chunk. (c) Update strategy: resident designs start with residual updates and switch
   once to covariance updates when a cost model with measured machine constants predicts a 25% saving for the
   coming level (src/chorus/costmodel.py); mode 'threshold' keeps the earlier rule for comparison. (d) Harness:
   exact peak working set of every process of the worker tree (GetProcessMemoryInfo on handles opened while
   the processes run) and device read bytes per disk; budget runs stop at M (not M + 0.5 GB).
   Invariance on tuning seeds (36 cases): identical stage-1 levels in all cases, candidate losses equal up to
   4e-5, and the frozen selection rule picks the same candidate for 576 of 576 responses. Accuracy runs made
   before this amendment are therefore kept. The timing plan of the revision is configs/timing_plan2.json
   (tools/make_timing_plan2.py), with tools/run_planner_grid.py, tools/run_patience.py and tools/build_store.py.
17. 2026-10-13, pooled relaxed refit (src/chorus/relax.py, Chorus(relax=True)), added after the review of
   the first accuracy draft showed that group-lasso shrinkage left Chorus behind gMCP on function error at
   SNR 4. Every candidate (GL and each (alpha, gamma), for each basis) is refit on its own support: least
   squares on the selected columns from the stage-2 Gram matrix and correlations (no extra read of the
   training rows; when stage 2 streams, one shared pass forms the Gram matrix of the union of supports),
   mixed with the penalized fit as phi * least squares + (1 - phi) * penalized, phi in {0, .25, .5, .75, 1}.
   One phi per candidate is chosen for all responses together by the validation error summed over responses
   in units of each response's validation variance. Supports, path entry order and the frozen selection rule
   are unchanged, so F1, MCC and AUPRC do not change. Design decided on tuning seeds 101-103 only
   (results/scratch/probe_*.py): a per-response guard (adopt when the cross-fitted gain exceeds kappa standard
   errors) lowered FEE by 4% and an empirical-Bayes version of that guard by 2.5%; the pooled weight lowered it
   by 4.6% (SNR 1: 0.0942 to 0.0901, SNR 4: 0.0323 to 0.0302; lower in 36 of 36 tuning cases). A
   validation-only check on the real cases (no test rows; phi chosen on one validation fold and scored on
   the other) gave Beijing 0.5964 to 0.6003, DepMap unchanged (phi = 0 on both folds). Engine check on the
   tuning seeds: supports and picks identical to the frozen run in 36 of 36 cases, penalized predictions equal
   up to 3e-6. All Chorus accuracy runs (simulation, robustness, sharing, real and confirmation cases, and the
   run without scale normalization) are repeated with this code; the earlier runs are kept as Chorus2_a16.
   Outputs add phi, pred_pen and per-candidate cand_pred_pen, so every selection ablation is reported with
   the same refit and a row without the refit is available.
18. 2026-10-13, start-up check of the memory budget (src/chorus/ledger.py preflight, called by the worker
   before any input is loaded). The timing run of the stress case at M = 0.5 GiB was stopped at 0.501 GiB
   during start-up, before the fit and its ledger had begun: the validation design (0.39 GiB) and the
   kernels had been loaded first. The ledger now adds the resident set before the import of chorus, a
   start-up allowance for the kernels (0.18 GiB, measured as 0.17 GiB in a fresh worker), the inputs the
   caller keeps resident and the fixed arrays of the fit, and refuses the budget before loading when the sum
   exceeds M - delta. For feasible budgets nothing changes (the check allocates nothing), and its decision
   agrees with the in-fit check: M = 0.5 GiB is refused for both budget workloads and M = 0.75 GiB for the
   stress case, M = 0.75 GiB is accepted for the n = 200,000 case. The two 0.5 GiB timing runs that ran
   before the change are repeated. Accuracy runs do not set a budget and are unaffected.
19. 2026-10-19, precision-matched external timing. The post hoc certificates show that adelie at its
   default tolerance returns group lasso solutions with relative duality gaps up to 4e-3 (DepMap) while
   Chorus certifies 1e-7. A probe on four DepMap responses gave gaps of 6.6e-3, 1.2e-3, 1.6e-4 and 2.0e-5
   for tolerances 1e-7, 1e-9, 1e-11 and 1e-13, at 2.6, 7.1, 20.7 and 54.2 seconds. adelie is therefore also
   timed at tolerance 1e-13 on every same-estimator workload (configs/timing_plan2_adelie_tight.json), and
   Table 2 reports both settings with their certified gaps.
20. 2026-10-20, reduced update-strategy grid. The registered grid (n up to 32,000, B up to 64, two repetitions)
   needed about one hour for the twelve cells at n = 500 and would have taken several days at n = 8,000 and
   32,000. It was stopped after the n = 500 cells (all B in {1, 16, 64}) and continues with n in
   {2,000, 8,000}, B in {1, 16} and one repetition per strategy (tools/run_planner_grid.py arguments).
   The paper reports the reduced grid and states the reduction.
   Figure 2i and its summary numbers use the 24 cells with B in {1, 16}. The three cells with B = 64 at
   n = 500 that ran before the reduction are reported separately in the text (largest ratio 1.66).

21. 2026-10-22, second review: statistics strategy, exit fairness, precision sweep, wide workloads.
   Written before any of the runs below. (a) Statistics strategy. When the Gram share of the ledger holds
   8D^2 bytes plus transients, stage 1 reads the design once (fused row-panel pass forming X'X/n and X'Y/n
   at fit time) and solves every response from these statistics with the certified Gram path of stage 2;
   the cost model chooses between statistics and working sets from the predicted number of passes,
   ceil(q/B)(1+K), and measured machine constants. strategy='working_set' reproduces the frozen engine
   bit for bit. The estimator, the selection rule and the tolerances are unchanged. Invariance check
   (tools/check_strategy_invariance.py) on the 36 tuning cases (seeds 101-103) plus Beijing and DepMap:
   if selected levels, picks and supports agree as in amendment 16, the accuracy runs stand; otherwise
   every Chorus accuracy run is repeated. The configuration is re-frozen after the check (the 10-20
   freeze is kept as configs/chorus_frozen_2026-10-20.json). (b) Exit fairness. skglm, Celer and adelie
   also run with the exit rule of Chorus (chorus.validation.ValTracker, the same folds, patience 5;
   adelie through grpnet exit_cond). Table 2 separates full paths from paths with exit. Exit losses are
   reported per response (validation loss, test R2, and F1 and FEE in simulation: median, 5th
   percentile, worst value, share below -0.01 in R2) from the existing patience outputs; DepMap adds one
   full-path run of the whole pipeline with a 10-hour cap. (c) Precision sweep. skglm and Celer at tol
   1e-4, 1e-6, 1e-8 (and 1e-10 on simulated workloads), adelie at 1e-4, 1e-7, 1e-9, 1e-11, 1e-13, Chorus
   at 1e-4, 1e-6, 1e-8, 1e-10, on q = 16, q = 64, Beijing and DepMap, with paths saved and every gap
   certified by tools/certify_external.py; 1-hour cap per run, a capped run is reported as not reached.
   Time to a gap target is the fastest setting whose largest certified gap meets it. (d) Repetitions.
   Every timing cell has three interleaved repetitions; tables report the median, the caption the count
   and the largest relative range. (e) Wide workloads. p = 5,000 and d = 5 (D = 25,000, full Gram 5 GB),
   n in {25,000, 50,000, 100,000}, q = 64, kappa = 0.5, the generator of the disk workloads otherwise
   unchanged, M = 2 GiB and a 3-hour cap; the budget sweep adds the wide n = 50,000 case. (f) Further
   executors. A full-Gram executor (one pass, statistics, no planner or ledger), a fixed-batch
   shared-scan executor (B = 64, fixed covariance updates, no ledger, no exit) and six-process parallel
   execution of adelie and skglm against six Chorus workers under the same total CPU.
   Addendum (same day, before generation): the wide cases use 2,000 validation and 2,000 test rows, because
   the resident validation design at 5,000 rows would take 1 GB of the 2 GiB budget; seed 2201, chunk 5,000.
   Addendum (same day): WorkingSet.to_gram wrote its product into the leading block of a wider buffer, which
   raises an error when the cost model switches to covariance updates with fewer than 256 columns; the product is
   now assigned instead. Runs that did not raise are unaffected; no recorded run had raised.
   Addendum (2026-10-24): from 10:00 on 10-23 another workload on the same machine (CPU at 94%) made every
   timing run 2 to 4.6 times slower; those runs (sweep repetitions 1 and 2, the first decomposition runs) and the DepMap
   Chorus run at 1e-6 that overlapped a short build of that workload are set aside in results/timing3_disturbed and
   repeated. From now on every timing run waits until other processes use less than 10% of the CPU for two
   minutes, the CPU time of all other processes is sampled during the run, and a run with more than 0.3 cores of
   background load on average is set aside and repeated.

22. 2026-10-28, pre-submission revision (author comments of 28 Oct), written before any of the runs below.
   (a) Baselines. The accuracy comparison uses six published methods, each named and cited once: GL (group
   lasso, Yuan and Lin 2006, Meier et al. 2009) from adelie paths at the default tolerance (results/*/adelieP*;
   skglm and Celer give the same estimator and appear only in the efficiency study), adaptive GL (Huang,
   Horowitz and Wei 2010) as a two-stage adelie fit (bench/py_worker.py method 'adaptive': stage 1 the
   validation-selected group lasso, stage 2 the groups it selected with penalties sqrt(d_j) ||b1_j||^(-gamma),
   gamma in {1, 2} and the stage-2 level by validation error; it replaces the version computed with the
   stage-2 solver of Chorus), GMCP (grpreg), SGL (sparsegl), SpAM (samQL, the quadratic-loss solver of SAM 1.3,
   with three and five functions) and MTL (biglasso). The approved plan named grpreg for adaptive GL; adelie is
   used instead because grpreg needs about 90 s per DepMap response, and adelie is a published solver of the
   same convex problem. GL with the 1-SE rule is no longer a baseline; it becomes the ablation row "selection
   by the 1-SE rule". The common protocol (basis size by validation, smallest validation error) is unchanged.
   (b) Matched model size on real data. Every baseline reruns on the six DepMap splits and the two Beijing
   origins with test predictions and active groups saved at every level (r_worker.R save_test_path,
   tools/run_real_a22.py, results/real_a22). For each response and method, the compared model is the path
   model with the smallest validation error among the models no larger than the model of Chorus (sizes in
   selected predictors, basis chosen by validation as in the protocol); its test R2 is compared with that
   of Chorus (tools/matched_size.py). Raw test R2 and model size stay in a table.
   (c) Tuning of Chorus, tuning data only. The 36 tuning cases of the main grid (seeds 101-103) plus 18 new
   tuning cases with seeds 201-202 in the settings of the robustness and sharing grids (base, n = 200, 12
   active, t3 noise, p = 500, mixed SNR, rescaled responses, two groups, q = 64 without sharing). No formal
   seed (1001+, 3001+, 4001+) and no real test row is used. Fit-level variants, each run once per tuning
   case with saved candidates: F0 the frozen configuration; F1 gamma in {0.5, 1, 2}; F2 bases d in {7, 5, 3};
   F3 relaxation weight phi chosen per response; F4 pooled phi on a finer grid of nine values; F5 alpha in
   {0, 0.5, 1}; F6 without scale-free norms; F7 patience 3; F8 patience 8. Selection variants applied offline to
   the saved candidates of every fit: zeta in {0.25, 0.5, 0.75, 1}, shrinkage eb, none or full pooling, and
   cross-fitted or plain validation losses. Decision rule, fixed now: the mean rank over tuning cases of
   F1, MCC, FEE and test R2 (as in amendment 14); fit-level variants are compared under the frozen
   selection rule, then the selection variants under the chosen fit; a variant is adopted only when its mean
   rank is lower than that of the frozen configuration by at least 0.1, otherwise the simpler choice is
   kept. A component (scale-free norms, cross-fitting, shrinkage) whose removal does not raise the mean rank
   by at least 0.1 is removed from the method. Patience also weighs fit time: a patience is adopted only when
   it does not raise the mean rank by more than 0.1 and lowers the time. The chosen configuration is frozen
   (configs/chorus_frozen_2026-10-2x.json, source hash) before any formal run.
   (d) Formal accuracy runs. Chorus with the frozen configuration and adaptive GL rerun on the 100 main,
   50 robustness and 55 sharing cases and the eight real cases; the other baselines keep their runs. The
   ablation is cumulative (GL; plus reweighting; plus borrowing across responses; plus the empirical Bayes
   selection; plus the choice of basis size; plus the relaxed refit), with one further row in which the 1-SE
   rule replaces the selection; standard errors over cases and a paired t-test at 5%. A sensitivity sweep on
   the formal cases (zeta, patience, gamma grid, basis sizes, relaxation mode) is reported only and changes
   nothing. Figure 3 shows the selections of one setting fixed now (kappa = 0.5, rho = 0.7, SNR 1) as
   selection frequencies over all 80 responses of its five seeds.
   (e) Short and wide designs. The working-set engine adds a per-response residual kernel for resident designs
   with n below the width of the candidate set, stops inner sweeps by the restricted duality gap, and no longer
   copies the Gram submatrix of a response at every call. The covariance kernel and the disk path are
   unchanged, so the out-of-core runs and Table 1 stand. Invariance check as in amendment 21 on the tuning
   cases, Beijing and 16 DepMap dependencies (identical levels, picks and supports; coefficients within 1e-7);
   if it fails the change is withdrawn. Then the Chorus cells of the precision sweep and of the decomposition
   are repeated (three repetitions, quiet gate); the external solvers keep their runs.
   Addendum (2026-10-28, after the tuning scores, before any formal run): the tuning scores
   (results/pilot/tune_a22_fit.csv, tune_a22_select.csv) favour F3 (mean rank 4.42) and F2 (4.49) over F0
   (5.05), which were never fitted together; removing scale-free norms raises the mean rank by 0.08 over
   all tuning cases (below the 0.1 bar) but by 0.72 on the robustness and sharing tuning cases. A
   combination check is added on the same tuning cases: F9 = F2 + F3, F10 = F9 without scale-free norms.
   The configuration with the lowest mean rank among F0, F2, F3, F9 and F10 (same rule and 0.1 margin) is
   chosen, then the selection variants are compared under it as written above.
   Decision (2026-10-28, authors, CP2): the combination check ranks F9 first (5.01), F10 5.35, F3 5.51, F2 5.60,
   F0 6.22. The authors freeze F3 (relaxation weight chosen per response, bases d in {5, 3}, everything else as
   frozen on 10-22) instead of F9, because F9 adds a d = 7 basis that no baseline receives under the common
   protocol; F9 gains 0.17 F1 points over F3 with equal FEE and R2 at twice the fit time. Scale-free norms,
   patience 5, the alpha and gamma grids and the selection rule (zeta 0.5, empirical Bayes shrinkage,
   cross-fitted losses) are kept. Frozen as configs/chorus_frozen_a22.json; the engine runs with switch_back
   off until the invariance check of (e).
   Addendum (2026-10-28, after the formal Chorus runs): the cumulative ablation depends on the order of its
   steps (scored on penalized fits, the selection step raises FEE before the refit, and borrowing chosen by
   plain validation error lowers F1 without shared predictors), so both designs were computed
   (tools/ablation_a22.py --order A, B, G, R and L). The paper reports the leave-one-out design (the layout
   of the tables of 10-13), the cumulative designs go to the technical report, and the 1-SE replacement is
   reported in the text. Results: results/v2/ablation_a22_*.csv.
