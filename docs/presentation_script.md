# Скрипт презентації: GNN-Guided Text Hallucination Repair

Цільова тривалість: **10-15 хвилин**

## Слайд 1: Назва

**Назва:** Виявлення та виправлення текстових галюцинацій у згенерованих зображеннях за допомогою GNN

Що сказати:

```text
Сьогодні я презентую підхід для виявлення та виправлення текстових галюцинацій у згенерованих зображеннях.

Проблема в тому, що сучасні image generation моделі вже можуть створювати реалістичні сцени,
але текст усередині цих сцен часто виходить неправильним:
літери міняються місцями, зникають, зливаються або стають нерозбірливими.

Наш підхід поєднує три компоненти:
OCR/LLM feedback, локальний refinement через inpainting, і GNN,
яка моделює геометрію тексту як граф.
```

Час: 45 секунд

## Слайд 2: Проблема

Що показати:

```text
згенероване зображення з поганим текстом
очікуваний текст
фактично згенерований неправильний текст
```

Що сказати:

```text
Важливо, що проблема не завжди в усьому зображенні.
Часто фон, освітлення, об'єкти і композиція виглядають нормально.
Помилка локалізована саме в зоні тексту.

Наприклад, модель може згенерувати правдоподібну вивіску,
але слово на ній буде з неправильними літерами або дивними штрихами.

Тому ми не хочемо перегенеровувати все зображення.
Ми хочемо знайти саме текстову область і виправити тільки її.
```

Час: 1 хвилина

## Слайд 3: Загальна ідея підходу

Що показати:

```text
RL feedback + refinement + GNN
```

Що сказати:

```text
Ми використовуємо три пов'язані ідеї.

Перша - RL-style feedback.
OCR і LLM-сигнали порівнюють згенерований текст з очікуваним ground truth.
Це дає reward або acceptance signal: прийняти результат, відхилити або спробувати refinement.

Друга - refinement.
Ми маскуємо тільки область з текстом і використовуємо inpainting,
щоб виправити цю локальну зону.
Так ми не руйнуємо решту фотографії.

Третя - GNN.
Ми представляємо символи або частини символів як вузли графа,
а просторові відношення між ними як ребра.
Після тренування GNN передбачає, які вузли неправильні,
як їх потрібно змістити, і як має виглядати repair mask.
```

Час: 1-1.5 хвилини

## Слайд 3.5: Модель генерації

Що показати:

```text
Base generator: Stable Diffusion / SDXL
Repair model: Stable Diffusion Inpainting
Our module: GNN mask + offset predictor
```

Що сказати:

```text
Як базову image generation модель ми беремо Stable Diffusion family.
У презентації можна формулювати як Stable Diffusion або SDXL.

Для етапу repair/refinement ми використовуємо Stable Diffusion Inpainting.
Тобто diffusion model не тренується з нуля.
Вона використовується як pretrained pixel-level repair module.

Наш внесок знаходиться перед inpainting:
ми додаємо GNN, яка прогнозує, де саме текст зіпсований,
яку область треба маскувати,
і які dx/dy зміщення потрібні для символів.
```

Коротко:

```text
Stable Diffusion генерує або ремонтує pixels.
GNN планує, де і як саме треба ремонтувати текст.
```

Час: 45 секунд

## Слайд 4: Архітектура

Що показати:

```text
Prompt + expected text
        |
Generated image
        |
OCR / synthetic labels
        |
Graph construction
        |
GNN correction model
        |
Mask + dx/dy offsets
        |
Inpainting refinement
        |
Validation
```

Що сказати:

```text
Це загальна архітектура системи.

На вході ми маємо prompt і очікуваний текст.
Далі є згенероване або синтетично зіпсоване зображення.

На етапі training ми знаємо bounding boxes з synthetic labels.
На реальних зображеннях ці boxes можуть приходити з OCR, наприклад EasyOCR або PaddleOCR.

Після цього ми будуємо граф:
вузли - це символи або компоненти символів,
ребра - це просторові відношення, сусідство і порядок читання.

GNN передбачає bad-node probability, dx/dy offsets і soft repair mask.
Потім ця маска передається в inpainting,
який виправляє тільки текстову область.

Фінально ми оцінюємо результат через OCR metrics, CLIP Score,
LPIPS, FID і геометричні метрики маски.
```

Час: 1 хвилина

## Після архітектури: що пояснити

Це ключовий момент презентації. Тут треба пояснити простими словами, чому саме GNN.

Рекомендований текст:

```text
Головна ідея в тому, що GNN не генерує пікселі.
Вона планує виправлення.

Текст має структуру.
Літери йдуть у певному порядку, мають схожий baseline,
між ними є відстані, вони належать до одного слова або одного рядка.

Звичайна OCR-маска бачить лише прямокутник:
"десь тут є текст".
Але вона не розуміє, що одна літера зсунута,
інша злилася з сусідньою, а третя взагалі неправильна.

Граф дозволяє явно змоделювати ці зв'язки.
Якщо символ зміщений або зіпсований,
його локальні відношення з сусідами змінюються.
GNN може навчитися знаходити такі порушення структури.

Після цього inpainting-модель виконує візуальну частину:
вона перемальовує тільки потрібну область,
але не змінює всю фотографію.
```

Коротка версія:

```text
GNN вирішує, де і як треба виправляти.
Inpainting виконує саме візуальне виправлення.
```

Час: 1 хвилина

## Слайд 5: Як текст стає графом

Що показати:

```text
text image -> character boxes -> graph nodes / graph edges
```

Що сказати:

```text
Після детекції тексту ми перетворюємо його на граф.

Кожен символ або connected component стає вузлом.
Для вузла ми зберігаємо числові ознаки:
нормалізовані x/y координати, ширину, висоту,
індекс символа в очікуваному тексті,
тип символа, наприклад літера або цифра,
і додаткові локальні ознаки.

Ребра будуються між сусідніми символами,
між символами в одному слові,
між найближчими spatial neighbors,
і між елементами, які лежать на одному рядку.

Для ребер ми зберігаємо relative dx/dy, distance,
кут, різницю в розмірах і sequence distance.

Тобто модель бачить не просто набір pixels,
а структуру тексту як об'єкт із відношеннями.
```

Технічна деталь, якщо є час:

```text
У коді graph_builder формує:
node_features: tensor [num_nodes, node_dim]
edge_index: tensor [2, num_edges]
edge_features: tensor [num_edges, edge_dim]

Це стандартне представлення для message passing у GNN.
```

Час: 1-1.5 хвилини

## Слайд 6: Що передбачає GNN

Що показати:

```text
node_error_probability
dx/dy offset
mask weight
inpaint strength
photo preservation weight
```

Що сказати:

```text
GNN має кілька виходів.

Перший вихід - ймовірність того, що вузол зіпсований.
Це binary classification на рівні символа або компонента.

Другий вихід - dx/dy offset.
Це регресія, яка каже, куди приблизно має зміститись символ або його область.

Третій вихід - mask weight.
Він показує, наскільки сильно ця область має входити в repair mask.

Також є region-level виходи:
inpainting strength і photo-preservation weight.
Вони потрібні, щоб не перемальовувати зайве.
```

Технічна деталь:

```text
У моделі використовується message passing.
Кожен вузол отримує повідомлення від сусідів через ребра.
Повідомлення залежить від ознак source node, destination node і edge features.
Після кількох шарів вузол має контекст не лише про себе,
а й про сусідні символи.
```

Час: 1-1.5 хвилини

## Слайд 7: Як формується repair mask

Що показати:

```text
bad nodes + offsets -> soft mask -> inpainting
```

Що сказати:

```text
Після GNN ми не одразу отримуємо фінальне зображення.
Ми отримуємо correction field.

Беремо символи з високою bad-node probability.
Для них беремо bounding boxes і predicted dx/dy offsets.
Потім формуємо маску, яка включає:
поточну неправильну позицію,
очікувану виправлену позицію,
невеликий padding навколо символів,
і blur по краях маски.

Це важливо, бо inpainting краще працює з м'якими масками,
а не з дуже різкими прямокутниками.
```

Технічна деталь:

```text
Функція prediction_to_mask бере boxes і outputs GNN,
порогує error/mask scores,
малює rectangles для source і predicted target positions,
а потім застосовує Gaussian blur.
```

Час: 1 хвилина

## Слайд 8: Dataset

Що показати:

```text
corrupted image
clean target image
repair mask
expected text
corrupted text
```

Що сказати:

```text
Для тренування ми використовуємо synthetic paired dataset.
Він синтетичний, але photo-like:
є вивіски, постери, labels, storefronts, window signs.

Кожен приклад має corrupted image і clean target image.
Також зберігаються expected text, corrupted text,
character boxes, target character boxes і repair mask.

Це дає supervised targets для GNN:
який вузол зіпсований,
який правильний offset,
і яку область потрібно відредагувати.
```

Час: 1 хвилина

## Слайд 9: Benchmark і складні кейси

Що показати:

```text
TextAtlas5 / TextAtlas5M
DrawBench
phone-like real-world stress
```

Що сказати:

```text
Ми не обмежуємось легкими прикладами.

Для TextAtlas5-style split ми моделюємо типові текстові помилки:
substitution, deletion, transpose, repeat і split/merge слів.

DrawBench-style split використовується для prompt following
і складніших текстових prompt-ів.

Окремо ми додали phone-like real-world stress:
handheld blur, glare, shadows, tiny labels,
cluttered packaging, occlusion і compression artifacts.

Це потрібно, щоб перевірити не тільки ідеальні synthetic samples,
а й умови, ближчі до реального світу.
```

Час: 1 хвилина

## Слайд 10: Training setup

Що показати:

```text
Train only GNN
Diffusion/inpainting model is pretrained
Loss = classification + offset regression + mask + preservation
```

Що сказати:

```text
Ми тренуємо тільки GNN.
Diffusion або inpainting модель не fine-tune-иться.

Вхід у тренування - це graph sample.
Таргети:
bad-node labels,
dx/dy offsets,
target boxes,
і repair mask.

Loss складається з кількох частин:
BCE для bad-node classification,
SmoothL1 для offset regression,
mask loss,
і preservation penalty,
який штрафує надмірне редагування фото.
```

Технічна деталь:

```text
На практиці:
error_logits -> BCEWithLogitsLoss
offset -> SmoothL1Loss
mask_logits -> BCEWithLogitsLoss
scale_rotation -> SmoothL1Loss до нульового target у прототипі
region_inpaint_strength / keep_photo_weight -> preservation loss
```

Що додати словами:

```text
Тренування швидке, бо модель легка,
а synthetic labels чисті.
Але для report-scale setup можна збільшити кількість samples
і запустити тренування на кілька годин.
```

Час: 1-1.5 хвилини

## Слайд 11: Validation metrics

Що показати:

```text
WA
OCR Accuracy
OCR F1
CER
CLIP Score
LPIPS
FID
mask IoU
offset MAE
```

Таблиця для слайду:

| Metric | Що вимірює | Бажаний напрям |
|---|---|---|
| WA | Частка повністю правильних слів | вище краще |
| OCR Accuracy | Exact match між OCR і expected text | вище краще |
| OCR F1 | Баланс precision/recall для розпізнаного тексту | вище краще |
| CER | Символьна edit distance / довжина expected text | нижче краще |
| Mask IoU | Перетин predicted mask з target mask | вище краще |
| Offset MAE | Середня помилка dx/dy зміщення | нижче краще |
| Node Error Accuracy | Точність пошуку пошкоджених вузлів | вище краще |
| CLIP Score | Відповідність image-text / prompt-image | вище краще |
| LPIPS outside mask | Наскільки змінилась область поза текстом | нижче краще |
| FID | Загальна реалістичність розподілу зображень | нижче краще |

Окрема таблиця текстових метрик для порівняння методів:

| Text metric | Формула / ідея | Що показує |
|---|---|---|
| Exact Match Accuracy | `OCR_text == expected_text` | чи весь текст повністю правильний |
| Word Accuracy | `correct_words / all_words` | скільки слів відновлено повністю |
| Character Accuracy | `1 - CER` | частка правильних символів |
| CER | `edit_distance(chars) / len(expected)` | кількість символьних помилок |
| Normalized Levenshtein Similarity | `1 - edit_distance / max_len` | наскільки близький OCR до ground truth |
| OCR F1 | character або token precision/recall | баланс пропущених і зайвих символів |
| Text Preservation Gain | `metric_after - metric_before` | наскільки repair покращив текст |

Для слайду з baseline comparison можна показувати:

| Method | WA ↑ | CER ↓ | Char Acc ↑ | Text Gain ↑ |
|---|---:|---:|---:|---:|
| Before repair | 0.3043 | 0.1557 | 0.8443 | - |
| Simple rectangular mask | очікувано краще за before | нижче за before | вище за before | середній |
| GNN + inpainting | upper bound 1.0000 | upper bound 0.0000 | upper bound 1.0000 | найбільший |

Важливо сказати:

```text
Для final чесного порівняння Simple baseline і GNN треба OCR після actual inpainting.
Зараз after-repair text metrics показані як synthetic clean-target upper bound.
Але сам набір метрик уже готовий для прямого порівняння.
```

Що сказати:

```text
Ми валідуємо систему з трьох сторін.

Перша - якість тексту:
Word Accuracy, OCR Accuracy, OCR F1 і Character Error Rate.

Друга - геометрія GNN:
mask IoU, offset MAE і node error accuracy.

Третя - якість зображення:
CLIP Score перевіряє alignment між prompt і image,
LPIPS outside mask перевіряє, чи не змінилась область навколо тексту,
FID оцінює загальну реалістичність.
```

Технічне пояснення:

```text
mask IoU показує, наскільки predicted mask збігається з target mask.
offset MAE показує середню помилку dx/dy.
CER - це edit distance на рівні символів, поділений на довжину expected text.
LPIPS рахується поза текстовою маскою, тому він саме про збереження фону.
```

Час: 1 хвилина

## Слайд 12: Results

Що показати:

```text
Main synthetic validation:
mask IoU: 0.6571
offset MAE: 0.0077
node error accuracy: 0.9964
CLIP before: 0.3039
CLIP after: 0.3143
LPIPS outside mask: 0.0132

TextAtlas-style stress:
mask IoU: 0.6253
offset MAE: 0.0081
node error accuracy: 0.9974
CLIP before: 0.2907
CLIP after: 0.3022
LPIPS outside mask: 0.0127
```

Таблиця результатів для слайду:

| Validation split | Mask IoU | Offset MAE | Node Acc | CLIP before | CLIP after | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Main synthetic | 0.6571 | 0.0077 | 0.9964 | 0.3039 | 0.3143 | 0.0132 |
| TextAtlas-style stress | 0.6253 | 0.0081 | 0.9974 | 0.2907 | 0.3022 | 0.0127 |

Таблиця text metrics:

| Split | WA before | CER before | WA upper bound | CER upper bound |
|---|---:|---:|---:|---:|
| Main synthetic | 0.3043 | 0.1557 | 1.0000 | 0.0000 |
| TextAtlas-style stress | 0.2222 | 0.1283 | 1.0000 | 0.0000 |

Що сказати:

```text
Результати показують, що GNN добре вивчає геометрію тексту.

Node error accuracy майже 1.0.
Це означає, що модель майже завжди правильно визначає,
які текстові вузли пошкоджені.

Offset error приблизно 0.008 normalized.
Для зображення 512 на 512 це приблизно 4 pixels.
Тобто модель досить точно прогнозує зміщення.

Mask IoU стабільний і на основній synthetic validation,
і на складнішому TextAtlas-style stress split.

LPIPS outside mask дуже низький,
тобто область поза текстом майже не змінюється.

Також CLIP Score після repair трохи вищий,
що означає кращий alignment з prompt або expected text.
```

Час: 1.5 хвилини

## Слайд 13: Baseline comparison

Що показати:

```text
Simple baseline:
  rectangular mask around observed corrupted boxes
  zero dx/dy offsets
  no graph reasoning

GNN:
  node-level corrupted text detection
  learned dx/dy offsets
  deformed soft repair mask
```

Таблиця для слайду:

| Method | Mask | Offset | Text structure |
|---|---|---|---|
| Simple baseline | прямокутник навколо corrupted boxes | zero offset | не використовує |
| GNN method | soft/deformed mask по вузлах | learned dx/dy | використовує граф |

Що сказати:

```text
Щоб показати, що GNN справді додає користь,
ми порівнюємо її з простим baseline.

Baseline дуже простий:
він бере один прямокутник навколо observed corrupted boxes,
не прогнозує зміщення,
і вважає, що всю область треба редагувати однаково.

Такий baseline може працювати на простих прикладах,
але він не розуміє структуру тексту.
Він не знає, яка літера неправильна,
куди її треба змістити,
і як зробити маску більш точною.

GNN, навпаки, працює на рівні вузлів.
Вона прогнозує bad-node probability,
dx/dy offsets і soft mask weights.
Тому repair mask стає більш локальною і структурною.
```

Технічна деталь:

```text
У validation script simple baseline рахується як:
simple rectangular mask навколо corrupted character boxes,
zero offset prediction,
all-bad node prediction.

Після цього ми порівнюємо:
GNN mask IoU vs simple mask IoU,
GNN offset MAE vs zero-offset MAE,
GNN node accuracy vs all-bad baseline.
```

Час: 1 хвилина

## Слайд 14: Важливе обмеження

Що показати:

```text
Current after-repair metrics = clean-target upper bound
Final evaluation = OCR on actual inpainted outputs
```

Що сказати:

```text
Важливе обмеження:
поточні after-repair text metrics використовують clean target як upper bound.

Це означає, що ми перевіряємо:
якщо refinement ідеально відновить правильний текст,
то метрики будуть такими.

Це доводить, що GNN правильно знаходить область і зміщення,
але це ще не повністю end-to-end OCR на actual inpainted images.

Наступний крок:
взяти predicted mask,
прогнати через Stable Diffusion inpainting,
потім запустити OCR на repaired image
і порахувати WA, OCR Accuracy, F1, CER, CLIP, LPIPS і FID вже на реальному output.
```

Час: 1 хвилина

## Слайд 15: Висновок

Що сказати:

```text
Підсумовуючи, ми розділяємо задачу на дві частини.

GNN відповідає за структуру тексту:
вона знаходить пошкоджені символи,
передбачає offsets,
і формує локальну repair mask.

Inpainting відповідає за pixels:
він виправляє тільки текстову область
і зберігає решту фото.

Валідація показує високу точність corrupted-node detection,
низьку offset error,
стабільний mask IoU,
низький LPIPS outside mask
і покращення CLIP Score.

Це показує, що GNN-guided refinement є перспективним напрямком
для виправлення текстових галюцинацій у generated images.
```

Час: 1 хвилина

## Таймінг

```text
Слайд 1: Назва                         0:45
Слайд 2: Проблема                      1:00
Слайд 3: Підхід                        1:30
Слайд 4: Архітектура                   1:00
Пояснення після архітектури            1:00
Слайд 5: Graph representation          1:15
Слайд 6: GNN outputs                   1:15
Слайд 7: Repair mask                   1:00
Слайд 8: Dataset                       1:00
Слайд 9: Benchmarks                    1:00
Слайд 10: Training                     1:15
Слайд 11: Metrics                      1:00
Слайд 12: Results                      1:30
Слайд 13: Baseline comparison          1:00
Слайд 14: Limitation                   1:00
Слайд 15: Conclusion                   1:00
```

Разом: приблизно **15 хвилин**.

## Коротка версія на випадок браку часу

```text
Метод виправляє text hallucination через поєднання OCR feedback,
GNN geometry prediction і local inpainting.

Ми представляємо символи як graph nodes,
а spatial relations як edges.
GNN передбачає, які символи неправильні,
як їх треба змістити,
і де має бути repair mask.

Після цього inpainting виправляє тільки текстову область,
не змінюючи решту зображення.

Тренується тільки GNN.
Diffusion модель залишається pretrained.

Валідація використовує WA, OCR Accuracy, OCR F1, CER,
CLIP Score, LPIPS, FID, mask IoU і offset MAE.

Результати показують високу node error accuracy,
низьку offset error,
стабільний mask IoU
і низький LPIPS outside mask.

Також ми порівнюємо метод із simple baseline:
звичайною rectangular mask без GNN і без offset prediction.
Це показує, що GNN додає структурне розуміння тексту.

Головне обмеження:
поточні after-repair text metrics є clean-target upper bound.
Наступний крок - OCR evaluation на actual inpainted outputs.
```
