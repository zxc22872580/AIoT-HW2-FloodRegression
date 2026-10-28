# 繳交指引：GPT 對話 PDF ＋ NotebookLM 摘要

這兩項作業要求必須用**你自己的帳號**操作，無法代勞。本檔提供可直接複製貼上的
prompt 與逐步流程。預估總時間：**40～60 分鐘**。

完成後執行這兩行即可更新繳交檔：

```bash
python make_report_pdf.py
python package_for_submission.py 7115056116
```

---

# 第一部分：GPT 對話 PDF

## 作業要求

> 所有與 ChatGPT 的對話請以 pdfCrowd 或其他方式須匯出為 PDF
> 請在報告中明確標示「GPT 輔助內容」

報告第 10 節已經寫好框架，你只需要補上 `GPT_conversation.pdf`。

## 步驟 A-1：開一個乾淨的新對話

重要：**開新對話**，不要用舊的聊天紀錄。匯出時會把整串對話都抓進去，
夾雜無關內容會讓老師不好評分。

## 步驟 A-2：依序提問

以下 8 個問題涵蓋本作業的實際決策點。**照順序問**，每題等 GPT 回完再問下一題。
不必把答案抄進報告——這份 PDF 的用途是證明 AI 輔助過程。

### Q1｜資料集選擇

```
我要做一份機器學習作業，要求從 Kaggle 找一個有 10 到 20 個特徵的公開資料集，
用多元線性回歸（Multiple Linear Regression）做預測，並按照 CRISP-DM 流程完成分析，
還要做特徵選擇和模型評估。請推薦 3 到 5 個適合的 Kaggle 資料集，
說明各自的特徵數、目標變數、優缺點，並告訴我哪一個最適合展示線性回歸。
```

### Q2｜CRISP-DM 架構

```
請詳細說明 CRISP-DM 的六個階段，每個階段在一份「多元線性回歸」報告中
具體應該寫哪些內容？另外請告訴我，學生在寫 CRISP-DM 報告時最常遺漏哪些項目？
```

### Q3｜特徵工程（這題最重要，務必問）

```
我的資料集有 20 個數值特徵，我發現這 20 個特徵的「逐列加總（row sum）」
和目標變數的相關係數高達 0.91，但每個單一特徵的相關係數只有 0.2 左右。
我應該把這個 row sum 當成一個新特徵加進「線性回歸」模型嗎？請說明理由。
```

> **為什麼這題最重要**：正確答案是**不應該**——`sum` 是 20 個原始特徵的精確線性組合，
> 對線性模型增加的資訊量恰好為零，還會讓設計矩陣奇異。
> 但很多 AI 會建議你加（因為這在**樹模型**上確實有效）。
>
> 如果 GPT 建議你加，**那正是報告第 10.2 節那張「AI 建議需人工修正」表格的真實佐證**，
> 價值極高，請務必保留這段對話。
> 如果 GPT 答對了，也很好——你可以追問下一題讓它說得更深入。

接著追問：

```
那如果我改用 XGBoost 這類樹模型，加入 row sum 會有幫助嗎？為什麼線性模型和樹模型
在這件事上的結論不同？
```

### Q4｜特徵選擇方法比較

```
請比較以下 5 種特徵選擇方法的原理、優缺點與適用時機：
SelectKBest (f_regression)、RFE、SequentialFeatureSelector、Lasso、Permutation Importance。
另外請問：如果這 5 種方法選出來的特徵彼此不一致，代表什麼意思？該怎麼解讀？
```

### Q5｜信賴區間 vs 預測區間

```
在線性回歸中，信賴區間（confidence interval）和預測區間（prediction interval）
有什麼差別？為什麼預測區間一定比信賴區間寬？
在 Python 的 statsmodels 裡要怎麼同時算出這兩種區間？
另外，我要怎麼驗證我算出來的 95% 預測區間是不是真的有 95% 的覆蓋率？
```

### Q6｜OLS 假設違反

```
我做完多元線性回歸後，Breusch-Pagan 檢定的 p 值是 1.8e-18（異質變異），
殘差偏態 +0.76、超峰度 +1.97（非常態）。
請問這兩項假設被違反會影響什麼、不影響什麼？
係數估計還能用嗎？有哪些補救方法？HC3 穩健標準誤是什麼，什麼時候該用？
```

### Q7｜線性模型 vs 樹模型

```
我的資料集上，多元線性回歸的測試 R² 是 0.84，但 HistGradientBoosting 只有 0.74
（訓練 R² 卻高達 0.97）。請問這代表什麼？
在什麼情況下線性模型會贏過梯度提升樹？我的訓練資料只有 4000 列，
這會不會是樹模型表現差的主因？
```

### Q8｜結果呈現

```
我要在報告中呈現多元線性回歸的結果，請問應該放哪些圖表最有說服力？
每種圖分別能說明什麼？另外評估指標應該報告哪些，
R² 和 Adjusted R² 的差別是什麼，什麼時候該看哪一個？
```

## 步驟 A-3：匯出成 PDF

### 方法一：pdfCrowd（作業指定，推薦）

1. 在 ChatGPT 對話頁面右上角點 **分享 / Share**
2. 點 **建立連結 / Create link**，複製產生的公開網址
   （必須是**公開連結**，pdfCrowd 才抓得到；私人對話它讀不到）
3. 前往 <https://pdfcrowd.com/web-to-pdf/>
4. 把連結貼進網址欄，按 **Convert**
5. 下載 PDF，檔名改成 **`GPT_conversation.pdf`**

### 方法二：瀏覽器列印（備援，最穩）

若 pdfCrowd 抓不到內容（分享連結是動態載入的，偶爾會失敗）：

1. 在對話頁面**先滑到最底**，確保所有訊息都已載入
   （ChatGPT 長對話是延遲載入的，沒滑到底會漏內容）
2. 按 `Ctrl + P`
3. 目的地選 **另存為 PDF**
4. 版面配置建議「直向」，勾選「背景圖形」才看得到對話框
5. 存成 **`GPT_conversation.pdf`**

### 步驟 A-4：放進資料夾

把 `GPT_conversation.pdf` 放到 **`claude file`** 資料夾（和 `REPORT.md` 同一層），
然後執行：

```bash
python package_for_submission.py 7115056116
```

打包腳本會自動偵測並收進 zip，待辦清單上的 GPT 項目也會消失。

---

# 第二部分：NotebookLM 研究摘要

## 作業要求

> 請使用 NotebookLM 對網路上同主題的解法進行研究，並撰寫一份 100 字以上的摘要
> 請在報告中明確標示「NotebookLM 摘要」

報告第 11 節目前是我寫的 320 字草稿（基於真實公開解法），
**你需要用 NotebookLM 自己產生一份來取代它**。

## 步驟 B-1：建立筆記本

1. 前往 <https://notebooklm.google.com/>，用 Google 帳號登入
2. 點 **建立新筆記本 / Create new**

## 步驟 B-2：加入來源

點 **新增來源 → 網站 / Website**，逐一貼上以下網址：

| # | 網址 | 內容 |
|---|---|---|
| 1 | `https://www.kaggle.com/competitions/playground-series-s4e5/writeups/aldparis-1st-place-solution` | 第 1 名解法（Ridge 集成） |
| 2 | `https://www.kaggle.com/competitions/playground-series-s4e5/writeups/peaky-blenders-2nd-place-solution-team-peaky-blend` | 第 2 名解法（LinearRegression 集成） |
| 3 | `https://www.kaggle.com/competitions/playground-series-s4e5/discussion/509042` | 第 3 名解法（Lasso 混合 57 模型） |
| 4 | `https://www.kaggle.com/competitions/playground-series-s4e5/writeups/4th-place-solution-hill-climbing-through-the-noise` | 第 4 名解法 |
| 5 | `https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499263` | 討論串：sum 公式 baseline |

> ### ⚠️ 若 NotebookLM 抓不到 Kaggle 頁面
>
> Kaggle 是動態網頁，NotebookLM 的網址擷取**有一定機率失敗**（會顯示來源是空的）。
> 備援做法：
>
> 1. 用瀏覽器開啟該頁面，`Ctrl + A` 全選 → `Ctrl + C` 複製
> 2. 回到 NotebookLM，**新增來源 → 複製的文字 / Pasted text**
> 3. 貼上並命名（例如「1st place solution」）
>
> 加入來源後，**先確認每個來源的摘要不是空白**，再進行下一步。

## 步驟 B-3：向 NotebookLM 提問

在對話框貼上這段（一次問完，它會綜合所有來源）：

```
請根據我提供的所有來源，分析 Kaggle Playground Series S4E5「洪水機率預測」競賽的
主流解法，並回答以下三個問題：

1. 這個資料集的目標變數與 20 個特徵之間，存在什麼樣的數學關係？
   公開討論中提到用最簡單的方法可以達到多少 R²？

2. 排名前四名的隊伍，各自使用什麼模型作為「最終模型」？
   樹模型和神經網路在他們的方案中扮演什麼角色？

3. 哪些特徵工程手法被證實有效？為什麼直接用 LightGBM 訓練原始特徵效果不好，
   但先扣除總和的線性項再對殘差建模就能大幅提升？

請以繁體中文撰寫一段約 300 字的連貫摘要，涵蓋上述三點，不要用條列式。
```

## 步驟 B-4：整理並驗證

1. 複製 NotebookLM 產生的摘要
2. **務必自己核對**：NotebookLM 會標註引用來源編號，點開確認它沒有編造。
   這份作業的報告反覆強調「AI 輔助但需人工驗證」，這一步就是實踐
3. 若字數不足 100 字，追問：`請把這段摘要擴充到 300 字，補充更多具體數據`

## 步驟 B-5：截圖存證

截下兩張圖（證明你真的用了 NotebookLM）：

- **截圖 1**：左側來源清單 + 你的提問
- **截圖 2**：NotebookLM 產生的摘要全文

存成 `notebooklm_01.png`、`notebooklm_02.png` 放進 `claude file` 資料夾。

## 步驟 B-6：取代報告第 11 節

1. 用記事本或 VS Code 開啟 **`REPORT.md`**
2. 找到 **`## 11. 【NotebookLM 研究摘要】`**
3. 把底下的「⚠️ 標示說明與待辦」區塊和「摘要草稿（約 320 字）」整段刪掉
4. 換成你的內容，格式如下：

```markdown
## 11. 【NotebookLM 研究摘要】

> ### 標示說明
> 本節內容由 Google NotebookLM 彙整以下 5 份公開來源後產生，並經人工核對。
> 使用截圖見 `notebooklm_01.png`、`notebooklm_02.png`。

**來源清單**：S4E5 競賽第 1～4 名解法 writeup，以及 sum baseline 討論串（連結見第 13 節）。

### 摘要

（在這裡貼上 NotebookLM 產生的摘要）

### 人工核對說明

本摘要經與第 9 節的獨立查證比對，內容一致／有以下差異：（如有差異請說明）
```

5. 存檔後執行：

```bash
python make_report_pdf.py
python package_for_submission.py 7115056116
```

---

# 完成檢查清單

- [ ] 用新對話問完 8 個 GPT 問題（Q3 務必問）
- [ ] 匯出 `GPT_conversation.pdf` 放進 `claude file` 資料夾
- [ ] NotebookLM 加入 5 個來源，確認來源不是空白
- [ ] 取得 300 字摘要並人工核對引用
- [ ] 存下 2 張 NotebookLM 截圖
- [ ] 取代 `REPORT.md` 第 11 節
- [ ] 執行 `python make_report_pdf.py`
- [ ] 執行 `python package_for_submission.py 7115056116`
- [ ] 確認待辦清單上 GPT 與 NotebookLM 兩項都已消失
- [ ] 上傳 `7115056116_hw2.zip`

---

# 對應評分標準

| 評分項目 | 佔比 | 對應內容 |
|---|---|---|
| CRISP-DM 流程完整且邏輯清楚 | 25% | 報告第 2～8 節 ✅ 已完成 |
| 包含 GPT 對話與 NotebookLM 摘要 | 15% | **本指引第一、二部分 ← 待你完成** |
| 有明確說明資料集來源與研究脈絡 | 10% | 報告第 1、9、13 節 ✅ 已完成 |
| 模型正確可執行，具特徵選擇與評估 | 25% | 主程式 + 報告第 5～7 節 ✅ 已完成 |
| 結果合理、美觀且具說服力 | 15% | 9 張圖 + 8 張表 ✅ 已完成 |
| Kaggle 名次／預測結果評估 | 10% | 報告第 7.1、7.4 節 ✅ 已完成 |
