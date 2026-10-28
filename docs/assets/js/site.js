/* =============================================================================
 * HW2 — Flood Regression Lab
 *
 * Renders every panel from assets/data/site_data.json, which tools/
 * export_site_data.py writes straight out of the fitted statsmodels OLS.
 *
 * The interactive predictor is not a lookup table: it rebuilds the 7 row
 * aggregates from the 20 sliders, applies the exported coefficient vector, and
 * derives the confidence and prediction intervals from the exported
 * coefficient covariance matrix with a real Student-t quantile. The numbers it
 * shows are the report's numbers, recomputed.
 *
 * Depends on: Chart.js
 * ========================================================================== */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
    );
  const icon = (id, style) =>
    '<svg class="ic"' + (style ? ' style="' + style + '"' : '') + '><use href="#' + id + '"/></svg>';

  const AGG_LABELS = {
    agg_std: '列內標準差',
    agg_max: '列內最大值',
    agg_min: '列內最小值',
    agg_median: '列內中位數',
    agg_skew: '列內偏態',
    agg_sum_sq: '總和二次項',
    agg_sum_cu: '總和三次項'
  };

  /* Chinese gloss for each of the 20 audit factors. */
  const FEATURE_ZH = {
    MonsoonIntensity: '季風強度',
    TopographyDrainage: '地形排水',
    RiverManagement: '河川治理',
    Deforestation: '森林砍伐',
    Urbanization: '都市化',
    ClimateChange: '氣候變遷',
    DamsQuality: '水壩品質',
    Siltation: '淤積',
    AgriculturalPractices: '農業行為',
    Encroachments: '違法佔用',
    IneffectiveDisasterPreparedness: '防災整備不足',
    DrainageSystems: '排水系統',
    CoastalVulnerability: '海岸脆弱度',
    Landslides: '山崩',
    Watersheds: '集水區',
    DeterioratingInfrastructure: '基礎設施老化',
    PopulationScore: '人口壓力',
    WetlandLoss: '濕地流失',
    InadequatePlanning: '規劃不當',
    PoliticalFactors: '政治因素'
  };

  const PHASES = [
    ['1', 'Business Understanding', '防災預算分配場景：以連續的洪水機率決定資源傾斜程度。論證為何此處需要「線性」模型 —— 係數可稽核、可轉為行動、運算成本低。設定 R² ≥ 0.8 為分析成功門檻。', 'business_understanding()'],
    ['2', 'Data Understanding', '資料品質總覽、目標分布與常態性、20 個特徵分布、相關係數矩陣。在此發現可加性結構，並載入原始資料集直接驗證生成規則。', 'data_understanding()'],
    ['3', 'Data Preparation', '缺失值檢查、離群值保留決策（高分是真實高風險區而非感測誤差）、7 個非線性聚合特徵、80/20 切分、僅以訓練集配適的標準化。', 'data_preparation()'],
    ['4', 'Modeling', '9 個模型於同一切分下比較：單純線性回歸、多元線性回歸、Ridge、Lasso、ElasticNet，並以樹模型作為對照組。', 'modeling()'],
    ['5', 'Evaluation', 'MAE／MSE／RMSE／R²／調整後 R²、5-fold 交叉驗證、OLS 五大古典假設檢定、HC3 穩健標準誤、信賴區間與預測區間。', 'evaluation()'],
    ['6', 'Deployment', '全量重訓、模型封存為 joblib、評分 API、Kaggle 提交檔產生，以及上線後的監控計畫。', 'deployment()']
  ];

  /* ---------------------------------------------------------------------------
   * Student-t quantile (needed for exact 95% bounds)
   * ------------------------------------------------------------------------ */

  const LANCZOS = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7
  ];

  function logGamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    const z = x - 1;
    let a = 0.99999999999980993;
    const t = z + 7.5;
    for (let i = 0; i < LANCZOS.length; i++) a += LANCZOS[i] / (z + i + 1);
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
  }

  /* Lentz continued fraction for the incomplete beta. */
  function betacf(a, b, x) {
    const FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-16) break;
    }
    return h;
  }

  function incompleteBeta(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(
      logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log1p(-x)
    );
    if (x < (a + 1) / (a + b + 2)) return (bt * betacf(a, b, x)) / a;
    return 1 - (bt * betacf(b, a, 1 - x)) / b;
  }

  function studentTCdf(t, df) {
    const half = incompleteBeta(df / 2, 0.5, df / (df + t * t)) / 2;
    return t >= 0 ? 1 - half : half;
  }

  /* Only ever called once per page load, so bisection is plenty. */
  function studentTInv(p, df) {
    let lo = -1e3, hi = 1e3;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (studentTCdf(mid, df) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /* ---------------------------------------------------------------------------
   * Formatting
   * ------------------------------------------------------------------------ */

  function fmt(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return v.toLocaleString('en-US', {
      minimumFractionDigits: d === undefined ? 4 : d,
      maximumFractionDigits: d === undefined ? 4 : d
    });
  }

  /* Coefficients span 1e-4 to 1e-1, so pick precision from magnitude. */
  function fmtCoef(v) {
    if (!isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1) return fmt(v, 4);
    if (a >= 1e-4) return v.toFixed(6);
    return v.toExponential(2);
  }

  function fmtP(p) {
    if (!isFinite(p)) return '—';
    if (p < 1e-4) return '< 0.0001';
    return p.toFixed(4);
  }

  /* 1.11e-16 reads better as a superscript than as a float. */
  function fmtSci(v, digits) {
    if (!isFinite(v)) return '—';
    const s = v.toExponential(digits === undefined ? 2 : digits);
    const m = s.match(/^(-?[\d.]+)e([+-])(\d+)$/);
    if (!m) return s;
    const sign = m[2] === '-' ? '−' : '';
    return m[1] + ' × 10<sup>' + sign + parseInt(m[3], 10) + '</sup>';
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ---------------------------------------------------------------------------
   * State
   * ------------------------------------------------------------------------ */

  const state = {
    data: null,
    inputs: {},
    charts: {},
    robustSE: false,
    tCrit: 1.96
  };

  /* ---------------------------------------------------------------------------
   * Prediction maths
   * ------------------------------------------------------------------------ */

  /* Rebuild the 7 aggregates exactly as add_aggregate_features() does:
   * pandas std/skew use ddof = 1 and the adjusted Fisher-Pearson coefficient. */
  function aggregates(vals) {
    const n = vals.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += vals[i];
    const mean = sum / n;

    let s2 = 0;
    for (let i = 0; i < n; i++) s2 += (vals[i] - mean) ** 2;
    const sd = Math.sqrt(s2 / (n - 1));

    const sorted = vals.slice().sort((a, b) => a - b);
    const mid = n >> 1;
    const median = n % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

    let skew = 0;
    if (sd > 0) {
      let m3 = 0;
      for (let i = 0; i < n; i++) m3 += ((vals[i] - mean) / sd) ** 3;
      skew = (n / ((n - 1) * (n - 2))) * m3;
    }

    const c = (sum - 100) / 10;
    return {
      agg_std: sd,
      agg_max: sorted[n - 1],
      agg_min: sorted[0],
      agg_median: median,
      agg_skew: skew,
      agg_sum_sq: c * c,
      agg_sum_cu: c * c * c,
      _sum: sum
    };
  }

  function designRow(vals) {
    const d = state.data;
    const agg = aggregates(vals);
    const x = [1];
    for (let i = 0; i < d.features.length; i++) x.push(vals[i]);
    for (let i = 0; i < d.aggregates.length; i++) x.push(agg[d.aggregates[i]]);
    return { x: x, sum: agg._sum };
  }

  function predict(vals) {
    const d = state.data;
    const row = designRow(vals);
    const x = row.x;
    const beta = d.model.params;

    let yhat = 0;
    for (let i = 0; i < x.length; i++) yhat += x[i] * beta[i];

    /* var(x0'b) = x0' V x0 */
    const V = d.model.cov;
    let varMean = 0;
    for (let i = 0; i < x.length; i++) {
      if (x[i] === 0) continue;
      let inner = 0;
      for (let j = 0; j < x.length; j++) inner += V[i][j] * x[j];
      varMean += x[i] * inner;
    }
    varMean = Math.max(varMean, 0);

    const seMean = Math.sqrt(varMean);
    const sePred = Math.sqrt(d.model.scale + varMean);
    const t = state.tCrit;

    return {
      yhat: yhat,
      sum: row.sum,
      ciLo: yhat - t * seMean,
      ciHi: yhat + t * seMean,
      piLo: yhat - t * sePred,
      piHi: yhat + t * sePred,
      seMean: seMean
    };
  }

  /* ---------------------------------------------------------------------------
   * Renderers
   * ------------------------------------------------------------------------ */

  function renderHero() {
    const d = state.data;
    const r = d.results;
    const ev = r.evaluation.test_metrics;
    const lb = r.deployment_leaderboard;

    const tags = [
      icon('i-layers') + ' <span>' + d.features.length + ' 個特徵 · ' +
        d.meta.n_rows_modelled.toLocaleString() + ' 列</span>',
      icon('i-chart') + ' <span>測試集 R² <b>' + ev.R2.toFixed(4) + '</b></span>',
      icon('i-trophy', 'color:#fcd34d') + ' <span>約第 <b>' + lb.rank.toLocaleString() +
        '</b> / ' + lb.n_teams.toLocaleString() + ' 名</span>',
      icon('i-shield', 'color:#6ee7b7') + ' <span>95% PI 實測覆蓋 <b>' +
        (r.prediction_intervals.empirical_pi_coverage * 100).toFixed(1) + '%</b></span>'
    ];
    $('heroTags').innerHTML = tags.map((t) => '<span class="hero-pill">' + t + '</span>').join('');

    $('dataNotice').hidden = false;
    $('dataNotice').innerHTML =
      icon('i-info') +
      '<div><strong>資料狀態：</strong>建模使用真實 Kaggle 競賽資料的 ' +
      d.meta.n_rows_modelled.toLocaleString() + ' 列公開鏡像抽樣（官方完整訓練集為 1,117,957 列，需登入下載）。' +
      '另取得原始 50,000 列資料集用於結構驗證，以及真實 private 排行榜用於名次定位。' +
      '訓練 ' + d.meta.n_train.toLocaleString() + ' 列 / 測試 ' + d.meta.n_test.toLocaleString() +
      ' 列，切分種子 ' + d.meta.random_state + '。</div>';

    $('footMeta').textContent =
      '最終模型：' + d.model.name + '　|　' + d.model.terms.length + ' 個參數　|　' +
      '殘差自由度 ' + d.model.df_resid.toLocaleString() + '　|　學號 ' + d.meta.student_id;
  }

  function renderMetrics() {
    const r = state.data.results;
    const ev = r.evaluation.test_metrics;
    const cards = [
      { t: '測試集 R²', v: ev.R2.toFixed(4), n: '解釋 ' + (ev.R2 * 100).toFixed(2) + '% 的樣本外變異',
        badge: ['badge-green', '達標 ≥ 0.8'], cls: 'good' },
      { t: 'RMSE', v: ev.RMSE.toFixed(5), n: '目標值標準差為 ' + r.data_understanding.target_stats.std.toFixed(4),
        badge: ['badge-blue', '越低越好'] },
      { t: 'MAE', v: ev.MAE.toFixed(5), n: '平均絕對誤差' },
      { t: '調整後 R²', v: ev.Adj_R2.toFixed(4), n: '已懲罰 ' + (state.data.model.terms.length - 1) + ' 個參數' },
      { t: '5-fold CV R²', v: r.modeling.cv5_r2_mean.toFixed(4),
        n: '± ' + r.modeling.cv5_r2_std.toFixed(4) + '（折間標準差）' },
      { t: '訓練／測試落差', v: r.evaluation.train_test_r2_gap.toFixed(5),
        n: '幾乎為零，無過度配適跡象', badge: ['badge-green', '健康'], cls: 'good' }
    ];
    $('metricsGrid').innerHTML = cards.map((c) =>
      '<div class="metric-card ' + (c.cls || '') + '">' +
      '<div class="metric-title"><span>' + esc(c.t) + '</span>' +
      (c.badge ? '<span class="badge-tag ' + c.badge[0] + '">' + esc(c.badge[1]) + '</span>' : '') +
      '</div><div class="metric-val">' + esc(c.v) + '</div>' +
      '<div class="metric-note">' + esc(c.n || '') + '</div></div>'
    ).join('');
  }

  function renderInsight() {
    const r = state.data.results;
    const du = r.data_understanding;
    const oc = du.original_dataset_check;

    $('kiSingle').textContent = du.max_abs_single_feature_corr.toFixed(4);
    $('kiSum').textContent = du.row_sum_corr_with_target.toFixed(4);
    $('kiDev').innerHTML = fmtSci(oc.max_abs_deviation_from_rule, 2);
    $('cmpSingle').textContent = du.max_abs_single_feature_corr.toFixed(4);
    $('cmpSum').textContent = du.row_sum_corr_with_target.toFixed(4);
    $('cmpRedundant').innerHTML = fmtSci(r.sum_redundancy_check.difference, 2);

    $('ruleBox').innerHTML =
      'FloodProbability = <span class="hl">0.005</span> × Σ(20 個因子)' +
      '<span style="color:var(--fg-faint)">　於 ' + oc.n_rows.toLocaleString() + ' 列原始資料上精確成立</span>';
  }

  function renderPhases() {
    $('phaseGrid').innerHTML = PHASES.map((p) =>
      '<div class="phase-card"><div class="num">PHASE ' + p[0] + '</div>' +
      '<h4>' + esc(p[1]) + '</h4><p>' + esc(p[2]) + '</p>' +
      '<span class="fn">' + esc(p[3]) + '</span></div>'
    ).join('');
  }

  function renderFeatureSelection() {
    const d = state.data;
    const fs = d.results.feature_selection;
    const rows = d.tables.selection_ranking.slice().sort((a, b) =>
      b.votes_out_of_5 - a.votes_out_of_5 || b.f_score - a.f_score);

    $('fsDecision').innerHTML = icon('i-check') + ' 決策：保留全部 ' + fs.best_k_by_holdout_r2 + ' 個特徵';

    const k10 = d.tables.k_sweep.find((r) => r.k === 10);
    $('fsWarning').innerHTML =
      '<strong>五種方法高度一致，但這個共識必須小心閱讀。</strong>' +
      '投票選出的 10 個特徵確實是「相對較強」的 10 個，然而 k-sweep 顯示 R² 隨 k <strong>嚴格遞增</strong>：' +
      '只留 10 個特徵時測試集 R² 僅 <b>' + (k10 ? k10.test_r2.toFixed(4) : '—') + '</b>，' +
      '保留全部 20 個才達到 <b>' + fs.all_20_r2.toFixed(4) + '</b>。' +
      '原因正是可加性結構 —— 每個因子都對總和有貢獻，刪掉任何一個都是直接丟失訊號。' +
      '因此本報告<strong>不採用</strong>特徵選擇的結果，而是保留全部特徵。';

    $('fsTableBody').innerHTML = rows.map((r) => {
      let dots = '';
      for (let i = 0; i < 5; i++) {
        dots += '<i class="vote-dot' + (i < r.votes_out_of_5 ? ' on' : '') + '"></i>';
      }
      return '<tr><td><strong>' + esc(r.feature) + '</strong>' +
        (FEATURE_ZH[r.feature] ? ' <span style="color:var(--fg-faint);font-size:11.5px">' +
          esc(FEATURE_ZH[r.feature]) + '</span>' : '') + '</td>' +
        '<td class="num">' + r.f_score.toFixed(2) + '</td>' +
        '<td class="num">' + r.rank_ftest + '</td>' +
        '<td class="num">' + r.rank_rfe + '</td>' +
        '<td>' + (r.sfs_selected ? '<span class="badge-tag badge-green">入選</span>'
          : '<span class="badge-tag badge-grey">—</span>') + '</td>' +
        '<td class="num">' + r.lasso_abs_coef.toFixed(5) + '</td>' +
        '<td class="num">' + r.perm_importance.toFixed(5) + '</td>' +
        '<td><span class="vote-dots">' + dots + '</span> <span class="num">' +
          r.votes_out_of_5 + '</span></td></tr>';
    }).join('');
  }

  function renderModels() {
    const rows = state.data.tables.model_comparison.slice()
      .sort((a, b) => b.test_R2 - a.test_R2);
    const finalName = state.data.results.evaluation.final_model;

    $('modelTableBody').innerHTML = rows.map((r) =>
      '<tr class="' + (r.model === finalName ? 'is-final' : '') + '">' +
      '<td><strong>' + esc(r.model) + '</strong>' +
      (r.model === finalName ? ' <span class="badge-tag badge-green">最終</span>' : '') + '</td>' +
      '<td class="num">' + r.n_features + '</td>' +
      '<td class="num">' + r.train_R2.toFixed(4) + '</td>' +
      '<td class="num" style="font-weight:600">' + r.test_R2.toFixed(4) + '</td>' +
      '<td class="num">' + r.test_Adj_R2.toFixed(4) + '</td>' +
      '<td class="num">' + r.test_RMSE.toFixed(5) + '</td>' +
      '<td class="num">' + r.test_MAE.toFixed(5) + '</td>' +
      '<td style="white-space:normal;max-width:34ch;color:var(--fg-faint);font-size:11.5px">' +
        esc(r.note || '') + '</td></tr>'
    ).join('');
  }

  function renderCoefficients() {
    const d = state.data;
    const coef = d.tables.coefficients;
    const robust = {};
    d.tables.robust_se.forEach((r) => { robust[r.term] = r; });
    const vif = {};
    d.tables.vif.forEach((r) => { vif[r.term] = r.VIF; });

    $('seHead').textContent = state.robustSE ? 'HC3 穩健標準誤' : '標準誤 (OLS)';
    $('pHead').textContent = state.robustSE ? 'p 值 (HC3)' : 'p 值 (OLS)';
    $('seToggleLabel').textContent = state.robustSE ? '切換回 OLS 標準誤' : '切換為 HC3 穩健標準誤';

    $('coefTableBody').innerHTML = coef.map((c) => {
      const rb = robust[c.term];
      const se = state.robustSE && rb ? rb.se_hc3 : c.std_err;
      const p = state.robustSE && rb ? rb.p_hc3 : c.p_value;
      const t = c.coef / se;
      const sig = p < 0.05;
      const isAgg = c.term.indexOf('agg_') === 0;
      const v = vif[c.term];

      let vifCell = '<span style="color:var(--fg-faint)">—</span>';
      if (v !== undefined && isFinite(v)) {
        const cls = v < 5 ? 'badge-green' : v < 10 ? 'badge-amber' : 'badge-red';
        const word = v < 5 ? '安全' : v < 10 ? '中等' : '共線';
        vifCell = '<span class="badge-tag ' + cls + '">' + v.toFixed(2) + ' ' + word + '</span>';
      }

      const label = c.term === 'const' ? '截距 (const)'
        : isAgg ? esc(c.term) + ' <span style="color:var(--fg-faint);font-size:11.5px">' +
            esc(AGG_LABELS[c.term] || '') + '</span>'
        : esc(c.term) + (FEATURE_ZH[c.term] ? ' <span style="color:var(--fg-faint);font-size:11.5px">' +
            esc(FEATURE_ZH[c.term]) + '</span>' : '');

      return '<tr><td><strong>' + label + '</strong></td>' +
        '<td class="num" style="font-weight:600">' + fmtCoef(c.coef) + '</td>' +
        '<td class="num">' + fmtCoef(se) + '</td>' +
        '<td class="num">' + t.toFixed(2) + '</td>' +
        '<td><span class="badge-tag ' + (sig ? 'badge-green' : 'badge-amber') + '">' +
          fmtP(p) + (sig ? '' : ' n.s.') + '</span></td>' +
        '<td class="ci">[' + fmtCoef(c.ci_low) + ', ' + fmtCoef(c.ci_high) + ']</td>' +
        '<td>' + vifCell + '</td></tr>';
    }).join('');

    const a = d.results.evaluation.assumptions;
    $('coefNote').innerHTML =
      '全部 20 個原始因子的係數都落在 <code>0.0055</code> 至 <code>0.0064</code> 的窄帶內，' +
      '與真實生成規則的 <code>0.005</code> 同一量級 —— 模型自行逼近了資料的生成機制。' +
      '7 個聚合特徵中只有 <code>agg_sum_cu</code> 與 <code>agg_max</code> 顯著，' +
      '顯示非線性修正確實存在但幅度有限。最大 VIF 為 <b>' + a.max_vif.toFixed(2) + '</b>，' +
      'VIF > 10 的項共 <b>' + a.n_terms_vif_gt_10 + '</b> 個，共線性在可接受範圍。' +
      '改用 HC3 穩健標準誤後，僅 <b>' + a.hc3_significance_flips + '</b> 項的顯著性判定改變。';
  }

  function renderAssumptions() {
    const a = state.data.results.evaluation.assumptions;
    const items = [
      {
        name: '1. 線性',
        pass: Math.abs(a.linearity_corr_fitted_resid) < 1e-6,
        stat: 'corr(配適值, 殘差) = ' + fmtSci(a.linearity_corr_fitted_resid, 2),
        why: '殘差與配適值正交，為 OLS 的代數必然結果；真正的線性檢查看殘差圖是否無系統性彎曲。'
      },
      {
        name: '2. 殘差獨立',
        pass: a.durbin_watson > 1.5 && a.durbin_watson < 2.5,
        stat: 'Durbin-Watson = ' + a.durbin_watson.toFixed(4),
        why: '極接近 2，無自我相關。各行政區為獨立觀測，本就不預期有序列相關。'
      },
      {
        name: '3. 同質變異',
        pass: a.homoscedasticity_ok,
        stat: 'Breusch-Pagan = ' + a.breusch_pagan_stat.toFixed(1) + '，p = ' + fmtSci(a.breusch_pagan_p, 2),
        why: '未通過。殘差變異隨配適值改變，故一律改用 HC3 穩健標準誤重新檢定顯著性。'
      },
      {
        name: '4. 殘差常態',
        pass: a.normality_ok,
        stat: 'Jarque-Bera p = ' + fmtSci(a.jarque_bera_p, 2) + '；偏態 ' +
              a.resid_skew.toFixed(4) + '，超額峰度 ' + a.resid_excess_kurtosis.toFixed(4),
        why: '未通過，殘差右偏。係數估計仍不偏，但名目 95% 預測區間的實際覆蓋率會略低於 95%。'
      },
      {
        name: '5. 無多重共線性',
        pass: a.n_terms_vif_gt_10 === 0,
        stat: '最大 VIF = ' + a.max_vif.toFixed(3) + '，VIF > 10 的項：' + a.n_terms_vif_gt_10 + ' 個',
        why: '通過。刻意不加入 agg_sum 與 agg_range 正是為了避免設計矩陣奇異。'
      }
    ];

    $('assumeGrid').innerHTML = items.map((it) =>
      '<div class="assume-card ' + (it.pass ? 'pass' : 'fail') + '">' +
      '<div class="head"><h4>' + esc(it.name) + '</h4>' +
      '<span class="badge-tag ' + (it.pass ? 'badge-green' : 'badge-amber') + '">' +
      icon(it.pass ? 'i-check' : 'i-warn') + (it.pass ? ' 通過' : ' 未通過') + '</span></div>' +
      '<div class="stat">' + it.stat + '</div>' +
      '<div class="why">' + esc(it.why) + '</div></div>'
    ).join('');
  }

  function renderIntervals() {
    const pi = state.data.results.prediction_intervals;
    $('piRatio').textContent = pi.pi_to_ci_width_ratio.toFixed(1);

    const rows = [
      ['95% 信賴區間 CI', '平均反應的估計誤差', pi.mean_ci_width, 0.95, pi.empirical_ci_coverage,
       '覆蓋率低是<strong>正確行為</strong> —— CI 本來就不是用來涵蓋個別觀測值的。'],
      ['95% 預測區間 PI', '單一新觀測值', pi.mean_pi_width, 0.95, pi.empirical_pi_coverage,
       '低估約 ' + ((0.95 - pi.empirical_pi_coverage) * 100).toFixed(1) + ' 個百分點，成因是殘差右偏。']
    ];

    $('intervalTableBody').innerHTML = rows.map((r) =>
      '<tr><td><strong>' + esc(r[0]) + '</strong></td>' +
      '<td>' + esc(r[1]) + '</td>' +
      '<td class="num">' + r[2].toFixed(5) + '</td>' +
      '<td class="num">95.0%</td>' +
      '<td class="num" style="font-weight:600">' + (r[4] * 100).toFixed(1) + '%</td>' +
      '<td style="white-space:normal;max-width:40ch;font-size:12px;color:var(--fg-muted)">' + r[5] + '</td></tr>'
    ).join('');

    $('intervalNote').innerHTML =
      'PI 平均寬度約為 CI 的 <strong>' + pi.pi_to_ci_width_ratio.toFixed(1) + ' 倍</strong>，' +
      '因此在圖中必須加上放大插圖才看得見 CI。' +
      '名目 95% 的 PI 實測只涵蓋 <strong>' + (pi.empirical_pi_coverage * 100).toFixed(2) + '%</strong>' +
      '（共 ' + pi.n_rows_for_coverage.toLocaleString() + ' 列），' +
      '本報告選擇<strong>實測並揭露</strong>這個落差，而不是只畫出一條看起來漂亮的色帶。';
  }

  function renderLeaderboard() {
    const lb = state.data.results.deployment_leaderboard;
    const dep = state.data.results.deployment;
    const boxes = [
      ['本模型測試集 R²', dep.best_linear_test_r2.toFixed(5), 'var(--accent)'],
      ['冠軍分數（' + lb.top_team + '）', lb.top_score.toFixed(5), 'var(--good)'],
      ['榜單中位數', lb.median_score.toFixed(5), ''],
      ['推估名次', '~' + lb.rank.toLocaleString() + ' / ' + lb.n_teams.toLocaleString(), ''],
      ['樹模型對照組 R²', dep.gbdt_test_r2.toFixed(5), 'var(--warn)']
    ];
    $('lbRow').innerHTML = boxes.map((b) =>
      '<div class="compare-box"><div class="lbl">' + esc(b[0]) + '</div>' +
      '<div class="big"' + (b[2] ? ' style="color:' + b[2] + '"' : '') + '>' + esc(b[1]) + '</div></div>'
    ).join('');

    $('lbCaveat').innerHTML =
      '此名次是以本模型的 hold-out R² 對照真實 private leaderboard 推估而得，<strong>並非實際提交結果</strong>。' +
      '榜單極度壓縮 —— 冠軍 ' + lb.top_score.toFixed(5) + ' 與中位數 ' + lb.median_score.toFixed(5) +
      ' 之間僅差 ' + ((lb.top_score - lb.median_score) * 1000).toFixed(1) + '‰，' +
      '數百個名次之間往往只隔幾個千分點，故名次僅供參考。' +
      '另需注意本模型僅以 ' + state.data.meta.n_rows_modelled.toLocaleString() +
      ' 列訓練，約為官方訓練集的 1/224。';
  }

  function renderGallery() {
    const figs = [
      ['04_key_insight_rowsum.png', '關鍵洞察：逐列總和 vs 目標', '單一特徵看似雜訊，20 個特徵的總和卻幾乎完全決定目標。這張圖是整份報告的轉折點。', true],
      ['09_prediction_with_intervals.png', '預測圖（含 CI 與 PI）', '四面板：實際 vs 預測、含信賴區間與預測區間的點預測、殘差圖、誤差分布。右上角插圖放大顯示被 PI 包住的 CI。', true],
      ['01_target_distribution.png', '目標分布與 Q-Q 圖', 'FloodProbability 的分布形狀與常態性檢查。'],
      ['02_feature_distributions.png', '20 個特徵分布', '全部為 0–16 的整數評分，分布形狀相近。'],
      ['03_correlation_heatmap.png', '相關係數矩陣', '特徵之間幾乎互不相關，與目標的個別相關也都很弱。'],
      ['05_feature_selection.png', '特徵選擇與 k-sweep 曲線', '五種方法的排名比較，以及 R² 隨特徵數嚴格遞增的證據。'],
      ['06_model_comparison.png', '模型比較', '9 個模型於同一切分下的表現。'],
      ['07_residual_diagnostics.png', '殘差診斷', 'OLS 五大假設的視覺檢查：殘差圖、Q-Q、尺度-位置、槓桿值。'],
      ['08_coefficients_ci.png', '係數與 95% 信賴區間', '20 個因子的係數緊密聚集，聚合特徵的區間多半跨越零。']
    ];
    $('galleryGrid').innerHTML = figs.map((f) =>
      '<figure class="fig-card' + (f[3] ? ' fig-wide' : '') + '" style="margin:0">' +
      '<img src="assets/figures/' + f[0] + '" alt="' + esc(f[1]) + '" loading="lazy">' +
      '<figcaption class="cap"><b>' + esc(f[1]) + '</b>' + esc(f[2]) + '</figcaption></figure>'
    ).join('');
  }

  /* Links point at the repository this page is published from, worked out from
   * the Pages URL, so renaming the repo does not break them. */
  function repoBase() {
    const h = location.hostname;
    const m = h.match(/^([^.]+)\.github\.io$/);
    if (!m) return null;
    const seg = location.pathname.split('/').filter(Boolean);
    if (!seg.length) return null;
    return 'https://github.com/' + m[1] + '/' + seg[0];
  }

  function renderDownloads() {
    const id = state.data.meta.student_id;
    const base = repoBase();
    const raw = (f) => (base ? base + '/raw/main/' + f : '../' + f);
    const items = [
      ['i-doc', '完整報告 PDF', '13 節 · CRISP-DM 全流程', raw(id + '_hw2_report.pdf')],
      ['i-code', '主程式 ' + id + '_hw2.py', '1,534 行 · 可直接執行', base ? base + '/blob/main/' + id + '_hw2.py' : '../' + id + '_hw2.py'],
      ['i-brain', 'GPT 對話紀錄', '已標示人工修正處', raw('GPT_conversation.pdf')],
      ['i-brain', 'NotebookLM 研究摘要', '含逐項人工核對', raw('NoteBookLM_pdf.pdf')],
      ['i-zip', '繳交壓縮檔', id + '_hw2.zip', raw(id + '_hw2.zip')],
      ['i-table', '所有數值結果 JSON', 'results.json · 報告數字唯一來源', base ? base + '/blob/main/outputs/results.json' : '../outputs/results.json']
    ];
    $('dlGrid').innerHTML = items.map((it) =>
      '<a class="dl-card" href="' + esc(it[3]) + '" target="_blank" rel="noopener">' +
      icon(it[0]) + '<span style="min-width:0"><span class="t">' + esc(it[1]) + '</span><br>' +
      '<span class="d">' + esc(it[2]) + '</span></span></a>'
    ).join('');

    if (base) $('repoLink').href = base;
  }

  /* ---------------------------------------------------------------------------
   * Charts
   * ------------------------------------------------------------------------ */

  function chartTheme() {
    return { grid: cssVar('--line'), text: cssVar('--fg-muted'), faint: cssVar('--fg-faint'), fg: cssVar('--fg') };
  }

  function drawCharts() {
    if (typeof Chart === 'undefined') return;
    const d = state.data;
    const t = chartTheme();
    Object.keys(state.charts).forEach((k) => state.charts[k] && state.charts[k].destroy());

    /* --- k-sweep --- */
    const ks = d.tables.k_sweep.slice().sort((a, b) => a.k - b.k);
    state.charts.k = new Chart($('kSweepChart').getContext('2d'), {
      type: 'line',
      data: {
        labels: ks.map((r) => r.k),
        datasets: [{
          label: '測試集 R²',
          data: ks.map((r) => r.test_r2),
          borderColor: cssVar('--accent'),
          backgroundColor: cssVar('--accent') + '22',
          fill: true,
          tension: 0.25,
          pointRadius: (c) => (c.dataIndex === ks.length - 1 || c.dataIndex === 9 ? 6 : 3),
          pointBackgroundColor: (c) =>
            c.dataIndex === ks.length - 1 ? cssVar('--good')
              : c.dataIndex === 9 ? cssVar('--warn') : cssVar('--accent')
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => 'R² = ' + c.parsed.y.toFixed(4),
              afterLabel: (c) => '加入：' + ks[c.dataIndex].added_feature
            }
          }
        },
        scales: {
          x: { title: { display: true, text: '特徵數 k', color: t.faint }, grid: { color: t.grid }, ticks: { color: t.text } },
          y: { title: { display: true, text: '測試集 R²', color: t.faint }, grid: { color: t.grid }, ticks: { color: t.text } }
        }
      }
    });

    /* --- votes --- */
    const votes = d.tables.selection_ranking.slice()
      .sort((a, b) => b.votes_out_of_5 - a.votes_out_of_5 || b.f_score - a.f_score);
    state.charts.v = new Chart($('votesChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: votes.map((r) => r.feature),
        datasets: [{
          label: '得票數',
          data: votes.map((r) => r.votes_out_of_5),
          backgroundColor: votes.map((r) =>
            r.votes_out_of_5 >= 3 ? cssVar('--accent') : cssVar('--line-strong')),
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { max: 5, grid: { color: t.grid }, ticks: { color: t.text, stepSize: 1 } },
          y: { grid: { display: false }, ticks: { color: t.text, font: { size: 10 } } }
        }
      }
    });

    /* --- model comparison --- */
    const mc = d.tables.model_comparison.slice().sort((a, b) => a.test_R2 - b.test_R2);
    const finalName = d.results.evaluation.final_model;
    state.charts.m = new Chart($('modelChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: mc.map((r) => r.model),
        datasets: [{
          label: '測試集 R²',
          data: mc.map((r) => r.test_R2),
          backgroundColor: mc.map((r) => (r.model === finalName ? cssVar('--good') : cssVar('--accent'))),
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => 'R² = ' + c.parsed.x.toFixed(5) } }
        },
        scales: {
          x: { grid: { color: t.grid }, ticks: { color: t.text } },
          y: { grid: { display: false }, ticks: { color: t.text, font: { size: 10 } } }
        }
      }
    });
  }

  /* ---------------------------------------------------------------------------
   * Predictor
   * ------------------------------------------------------------------------ */

  function buildSliders() {
    const d = state.data;
    $('sliderBox').innerHTML = d.features.map((f, i) => {
      const s = d.feature_stats[f];
      const start = Math.round(s.median);
      state.inputs[i] = start;
      return '<div class="slider-row">' +
        '<div class="lbl"><span>' + esc(f) +
        (FEATURE_ZH[f] ? ' <span style="color:var(--fg-faint);font-weight:500">' + esc(FEATURE_ZH[f]) + '</span>' : '') +
        '</span><span class="val" id="sv-' + i + '">' + start + '</span></div>' +
        '<input type="range" id="sl-' + i + '" min="0" max="' + Math.max(16, s.max) +
        '" step="1" value="' + start + '">' +
        '</div>';
    }).join('');

    d.features.forEach((f, i) => {
      $('sl-' + i).addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        state.inputs[i] = v;
        $('sv-' + i).textContent = v;
        updatePrediction();
      });
    });

    /* Preset levels are chosen so the row sum lands inside the range the model
     * was fitted on (see the extrapolation warning): 4/5/6 give sums of
     * 80/100/120 against a training range of roughly 67-132. */
    const presets = [
      ['資料集中位數', (f) => Math.round(state.data.feature_stats[f].median)],
      ['偏低風險', () => 4],
      ['偏高風險', () => 6],
      ['隨機抽樣', (f) => {
        const s = state.data.feature_stats[f];
        return Math.round(s.mean + (Math.random() * 2 - 1) * s.std * 1.5);
      }]
    ];
    $('presetRow').innerHTML = presets.map((p, i) =>
      '<button type="button" class="preset-btn" data-preset="' + i + '">' + esc(p[0]) + '</button>'
    ).join('');

    $('presetRow').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-preset]');
      if (!btn) return;
      const fn = presets[parseInt(btn.dataset.preset, 10)][1];
      d.features.forEach((f, i) => {
        const s = d.feature_stats[f];
        const v = Math.min(Math.max(fn(f), 0), Math.max(16, s.max));
        state.inputs[i] = v;
        $('sl-' + i).value = v;
        $('sv-' + i).textContent = v;
      });
      updatePrediction();
    });
  }

  function updatePrediction() {
    const d = state.data;
    const vals = d.features.map((f, i) => state.inputs[i]);
    const p = predict(vals);

    $('predValue').textContent = p.yhat.toFixed(4);
    $('predSub').textContent =
      'CI [' + p.ciLo.toFixed(4) + ', ' + p.ciHi.toFixed(4) + ']　' +
      'PI [' + p.piLo.toFixed(4) + ', ' + p.piHi.toFixed(4) + ']';

    /* Draw both bands on one axis, padded so the PI never touches the edge. */
    const pad = (p.piHi - p.piLo) * 0.12;
    const lo = p.piLo - pad;
    const hi = p.piHi + pad;
    const span = hi - lo || 1;
    const pct = (v) => ((v - lo) / span) * 100;

    $('bandPI').style.left = pct(p.piLo) + '%';
    $('bandPI').style.width = (pct(p.piHi) - pct(p.piLo)) + '%';
    $('bandCI').style.left = pct(p.ciLo) + '%';
    $('bandCI').style.width = Math.max(pct(p.ciHi) - pct(p.ciLo), 0.4) + '%';
    $('bandPoint').style.left = pct(p.yhat) + '%';
    $('scaleLo').textContent = lo.toFixed(3);
    $('scaleHi').textContent = hi.toFixed(3);

    $('sumVal').textContent = p.sum.toFixed(0);
    $('naiveVal').textContent = (0.005 * p.sum).toFixed(4);

    /* agg_sum_cu grows with the cube of the centred row sum, so a sum well
     * outside the fitted range drags the prediction somewhere meaningless.
     * Say so rather than letting the number stand. */
    const rs = d.row_sum;
    const out = p.sum < rs.min || p.sum > rs.max;
    $('extrapWarn').hidden = !out;
    if (out) {
      $('extrapWarn').innerHTML =
        icon('i-warn') +
        '<div><strong>已超出訓練範圍，此預測不可信。</strong>目前總和為 ' + p.sum.toFixed(0) +
        '，但模型只在總和 ' + rs.min.toFixed(0) + '–' + rs.max.toFixed(0) +
        ' 的資料上配適過。由於模型含有總和的二次與三次項，' +
        '在範圍外會急速發散 —— 這正是多項式項的代價，也是外推必須謹慎的實例。</div>';
    }
  }

  /* ---------------------------------------------------------------------------
   * Boot
   * ------------------------------------------------------------------------ */

  function bindChrome() {
    $('themeToggle').addEventListener('click', () => {
      const root = document.documentElement;
      const explicit = root.getAttribute('data-theme');
      const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const now = explicit || (systemDark ? 'dark' : 'light');
      root.setAttribute('data-theme', now === 'dark' ? 'light' : 'dark');
      try { localStorage.setItem('hw2-theme', root.getAttribute('data-theme')); } catch (e) { /* blocked storage */ }
      if (state.data) drawCharts();
    });

    $('seToggle').addEventListener('click', () => {
      state.robustSE = !state.robustSE;
      renderCoefficients();
    });

    $('galleryGrid').addEventListener('click', (e) => {
      const img = e.target.closest('img');
      if (!img) return;
      $('lightboxImg').src = img.src;
      $('lightboxImg').alt = img.alt;
      $('lightbox').hidden = false;
    });
    $('lightbox').addEventListener('click', () => { $('lightbox').hidden = true; });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') $('lightbox').hidden = true;
    });

    try {
      const saved = localStorage.getItem('hw2-theme');
      if (saved === 'dark' || saved === 'light') {
        document.documentElement.setAttribute('data-theme', saved);
      }
    } catch (e) { /* blocked storage */ }
  }

  function boot() {
    bindChrome();
    fetch('assets/data/site_data.json')
      .then((r) => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then((data) => {
        state.data = data;
        state.tCrit = studentTInv(0.975, data.model.df_resid);
        renderHero();
        renderMetrics();
        renderInsight();
        renderPhases();
        renderFeatureSelection();
        renderModels();
        renderCoefficients();
        renderAssumptions();
        renderIntervals();
        renderLeaderboard();
        renderGallery();
        renderDownloads();
        $('sumRange').textContent =
          '（訓練範圍 ' + data.row_sum.min.toFixed(0) + '–' + data.row_sum.max.toFixed(0) + '）';
        buildSliders();
        updatePrediction();
        drawCharts();
        window.__siteReady = true;
      })
      .catch((err) => {
        $('dataNotice').hidden = false;
        $('dataNotice').className = 'notice warn';
        $('dataNotice').innerHTML =
          icon('i-warn') + '<div>無法載入 <code>assets/data/site_data.json</code>（' +
          esc(err.message) + '）。請先執行 <code>python tools/export_site_data.py</code>，' +
          '並以 HTTP 伺服器開啟本頁（直接開啟本機檔案會被瀏覽器的同源政策擋下）。</div>';
        window.__siteReady = false;
      });
  }

  boot();
})();
