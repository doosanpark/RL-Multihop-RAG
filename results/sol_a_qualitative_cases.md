# Solution A — 정성 사례 분석 (SFT vs RL seed42, val[0:200])

## 통계 요약 (paired, n=200)

- mean F1: SFT **0.402** → RL **0.464** (Δ=+0.062)
- RL이 SFT를 큰 폭(Δ>0.5)으로 고친 케이스: **19**
- RL이 망친 케이스(Δ<-0.5): **9**
- 둘 다 실패(F1=0): **82** (← 0.5B 천장)
- 타입별 평균 Δ: comparison +0.026, bridge +0.069 (RL 이득은 comparison에 집중 — 정량과 일치)

## 사례 1 — RL이 SFT를 고친 비교 추론 (Top-3 by Δ)

### Case 1.1 (bridge)
- **Q**: A Japanese manga series based on a 16 year old high school student Ichitaka Seto, is written and illustrated by someone born in what year?
- **type**: bridge | **gold**: `1962`
- **SFT pred**: `1974` (F1=0.00)
- **RL pred**: `1962` (F1=1.00) — **Δ=+1.00**


```
SFT gen (마지막 200자):
e entire student body. The series has been licensed for an English release by Seven Seas Entertainment. An anime television adaptation by Studio Deen aired between April 8, 2016 and July 1, 2016.
</information>
<think>Combining the retrieved evidence, the answer is 1974.</think><answer>1974</answer>

RL gen (마지막 200자):
he way. Things become even more complicated when Itsuki Akiba returns to Japan; she is a girl Ichitaka was friends with in their childhood before she moved to the United States, and who had a ...
</information>
<think>Combining the retrieved evidence, the answer is 1962.</think><answer>1962</answer>
```

### Case 1.2 (bridge)
- **Q**: How many copies of Roald Dahl's variation on a popular anecdote sold?
- **type**: bridge | **gold**: `250 million`
- **SFT pred**: `25` (F1=0.00)
- **RL pred**: `250 million` (F1=1.00) — **Δ=+1.00**


```
SFT gen (마지막 200자):
morous mink coat from a man with whom she had an affair. She hopes to sneak the coat into her home without arousing her husband's suspicions, but soon discovers that her husband has plans of his own.
</information>
<think>Combining the retrieved evidence, the answer is 25.</think><answer>25</answer>

RL gen (마지막 200자):
rom a man with whom she had an affair. She hopes to sneak the coat into her home without arousing her husband's suspicions, but soon discovers that her husband has plans of his own.
</information>
<think>Combining the retrieved evidence, the answer is 250 million.</think><answer>250 million</answer>
```

### Case 1.3 (bridge)
- **Q**: What is the name of the detective novelist parent of the author of '"Q" Is for Quarry'?
- **type**: bridge | **gold**: `C. W. Grafton`
- **SFT pred**: `Kinsey Millhone` (F1=0.00)
- **RL pred**: `C. W. Grafton` (F1=1.00) — **Δ=+1.00**


```
SFT gen (마지막 200자):
six million tonnes of granite were excavated from the quarry, giving Aberdeen the name of 'The Granite City'. Rubislaw Quarry is one of the biggest man-made holes in Europe.
</information>
<think>Combining the retrieved evidence, the answer is Kinsey Millhone.</think><answer>Kinsey Millhone</answer>

RL gen (마지막 200자):
r Quarry: "Q" Is for Quarry is the 17th novel in Sue Grafton's "Alphabet" series of mystery novels and features Kinsey Millhone, a private eye based in Santa Teresa, California.
</information>
<think>Combining the retrieved evidence, the answer is C. W. Grafton.</think><answer>C. W. Grafton</answer>
```

## 사례 2 — 둘 다 실패한 bridge (천장의 정체)

### Case 2.1 (bridge, 둘 다 F1=0)
- **Q**: What football club plays in the area between the old tool gates: Brook Bar and Trafford bar?
- **type**: bridge | **gold**: `Manchester United`
- **SFT pred**: `Ragbi Klub Mornar Bar` (F1=0.00)
- **RL pred**: `Ragbi Klub Mornar Bar` (F1=0.00) — **Δ=+0.00**


```
SFT pred: Ragbi Klub Mornar Bar
RL pred:  Ragbi Klub Mornar Bar
gold:     Manchester United
```

### Case 2.2 (bridge, 둘 다 F1=0)
- **Q**: Alvaro Mexia had a diplomatic mission with which tribe of indigenous people?
- **type**: bridge | **gold**: `Apalachees`
- **SFT pred**: `Ais` (F1=0.00)
- **RL pred**: `Ais` (F1=0.00) — **Δ=+0.00**


```
SFT pred: Ais
RL pred:  Ais
gold:     Apalachees
```

## 사례 3 — RL이 오히려 망친 케이스 (정직성)

### Case 3.1 (bridge)
- **Q**: In which year was the King who made the 1925 Birthday Honours born?
- **type**: bridge | **gold**: `1865`
- **SFT pred**: `1865` (F1=1.00)
- **RL pred**: `1865–1866` (F1=0.00) — **Δ=-1.00**

### Case 3.2 (comparison)
- **Q**: Who died first, George Archainbaud or Ralph Murphy?
- **type**: comparison | **gold**: `George Archainbaud`
- **SFT pred**: `George Archainbaud` (F1=1.00)
- **RL pred**: `Ralph Murphy` (F1=0.00) — **Δ=-1.00**
