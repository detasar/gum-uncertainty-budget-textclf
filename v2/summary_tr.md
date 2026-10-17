# Özet (v2, ara rapor, 17 Ekim 2026; API-LLM bölümü 18:10 civarında eklendi)

Bu çalışma, sıfır-atışlı (zero-shot) metin sınıflandırmada konformal tahmin kümelerinin ortalama boyutu için GUM tarzı bir belirsizlik bütçesini yeniden sınıyor. v1'in iddiaları, `fab3c8a` commit'iyle önceden kaydedilen (preregistration) bir tasarımla test edildi. Tasarımın unsurları şunlar:
- geçerliliği iki bağımsız yargıç modelle denetlenmiş 30 rastgele etiket ifadesi seti,
- 200 rastgele kalibrasyon/test bölünmesi,
- iki yönlü rastgele etkiler modeli,
- beş veri seti (AG News, BBC 2024-25, SST-2, DBpedia-14, MASSIVE-18).

Sonuçlar ara niteliktedir. Önceden kayıtlı 17 sıfır-atışlı hücrenin 15'i bi-encoder, biri NLI (deberta-v3-small), biri Qwen2.5-0.5B; NLI ve Qwen hücreleri yalnızca AG News'e ait. Bunlara ek olarak, önceden kayıtlı olmayan 10 keşifsel API-LLM hücresi (T5) var: gpt-6-luna ve gpt-4.1-nano, beş veri setinin hepsinde. Modeller yalnızca kendi türleri içinde karşılaştırıldı. Üretken LLM'ler (yerel T3 ve API T5) yalnızca birbiriyle karşılaştırıldı; deterministik NLI ve bi-encoder sınıflandırıcılara karşı sıralanmadı.

**Ana bulgular (önceden kayıtlı hücreler)**
- **Büyüklük.** n = 250'de etiket ifadesi bileşeni 17 hücrenin tamamında sıfırdan farklı. Bi-encoder'larda medyan σ_ifade/σ_örnekleme oranı 2,94, medyan ifade payı 0,75 (aralık 0,31–0,88). Pay, NLI hücresinde 0,82, Qwen hücresinde 0,92.
- **Taban.** İfade bileşeni n ile küçülmüyor; eğim −0,04 ile 0,03 arasında. Örnekleme bileşeni ise −0,35 ile −0,87 arasındaki eğimlerle düşüyor. W = 4 alt kümeleri, W = 30 tahmininin 0,87–0,96'sını veriyor.
- **Kapsama.** Aynı ifadeyle kalibre edildiğinde ifade, kapsamayı etkilemiyor (pay 0,00). Kalibrasyondan sonra ifade değişirse ortalama kapsama 0,732'ye kadar düşüyor (hedef 0,90).
- **Anlam mesafesi.** Bir etiketin anlamca uzaklığı tahmin kaymasını öngörüyor (bi-encoder ρ = 0,56). Yalnızca büyük/küçük harf değişikliği, geçerli parafraz kaymasının NLI'da 0,60'ını, Qwen'de 0,35'ini üretiyor.
- **Tekrarlanabilirlik (yerel CPU).** Aynı fp32 koşullarındaki dört bağımsız çalıştırmada üç araç türünde de max |Δp| = 0. fp32'de batch boyutu ya da iş parçacığı sayısı değişince fark ≤ 3e-6. bf16 kullanımı ise NLI ve bi-encoder'da konformal kümelerin 0,01–0,11'ini değiştiriyor (H5b, T1 ve T2 için destekleniyor).

**Keşifsel API-LLM bulguları (T5; önceden kayıttan sapma)**
- **Sapma notu.** Yerel LLM hücreleri CPU sınırları nedeniyle çok yavaş ilerlediği için, LLM kanıtı 17 Ekim 2026'da (17:10–17:58, İstanbul saati, UTC+3) iki kapalı API modeliyle toplandı: `gpt-6-luna` (Decisions uç noktası, seçenek başına olasılık) ve `gpt-4.1-nano-2025-04-14` (chat completions, ilk-token harf logprob'ları, top-20). Kapalı modeller aynı ad altında sessizce değişebilir; bu sonuçlar o tarih ve o model kimlikleri için bir anlık görüntüdür.
- **Bütçe.** 10 API hücresinin hepsinde ifade baskın: n = 250'de ifade payı 0,75–0,96, σ_ifade/σ_örnekleme oranı 3,17–9,62 (10 hücrenin 10'unda > 3). Bölünme örneklemesinin payı 0,010–0,074. İfade bileşeni n ile küçülmüyor (eğim −0,08…0,01; n_max'ta GA 10/10 hücrede 0'ı dışlıyor).
- **Seçenek sırası, yeni bir belirsizlik kaynağı.** AG News ve BBC'de seçenekler iki ek rastgele sırayla da soruldu. Küme boyutu varyansının payları şöyle: AG gpt-6-luna ifade 0,81 / sıra 0,11 / örnekleme 0,007; AG gpt-4.1-nano 0,78 / 0,16 / 0,010; BBC gpt-6-luna 0,72 / 0,13 / 0,031; BBC gpt-4.1-nano 0,77 / 0,19 / 0,006. Sıra değiştirmek, ifade değiştirmenin yaklaşık yarısı kadar tahmini çeviriyor (0,043–0,123'e karşı 0,095–0,226).
- **gpt-6-luna.** Olasılıklar 0,01'lik ızgaraya nicemlenmiş (101 farklı değer; pilotta %62,5'i tam 0, %17,3'ü tam 1). Doğru sınıfın olasılığı maddelerin α'dan fazlasında 0 olduğunda LAC eşiği bu bağa düşüyor ve kümeler neredeyse tüm K etikete sıçrıyor. AG'de iki geçerli sette ortalama küme boyutu K'nın 1,00'i ve 0,95'i (bu setlerde 0'daki pay %16,6 ve %13,9). Bu sıçramalar AG'deki büyük σ_ifade değerini (0,727 [0,233, 1,043]) şişiriyor. Ret ("refusal") yanıtları: AG 347/140.400, BBC 826/140.400, DBpedia 251/47.000, MASSIVE 682/56.400, SST-2 0. Pilotta istek başına 1, 8 veya 31 soru sormak ve yeniden çalıştırmak bit düzeyinde aynı sonucu verdi. Üç çalıştırmalık test-tekrar testinde max |Δp| = 0 çıktı.
- **gpt-4.1-nano.** Top-20 dışında kalan seçenek harflerinin oranı (bunlara üst sınır olarak en küçük döndürülen logprob atanıyor) SST-2'de 0,077, DBpedia'da 0,649. Tek oturumda 8 farklı `system_fingerprint` görüldü. Sıcaklık 0'da üç çalıştırma arasında max |Δp| = 0,927 oldu, maddelerin %37,9'unda p değişti ve LAC kümeleri test maddelerinin %3,6'sında (bir sette %15'e kadar) değişti. **Bu önemli bir bulgu: API modelleri için tekrarlanabilirliğin sıfır olduğu varsayımı kısmen çürüdü.**
- **Toplam API harcaması:** 11,24 $ (üst sınır 23 $; liste fiyatları, önbellek indirimi hesaba katılmadan).

**v1 iddialarına hükümler**
- "%32 ifade / %64 örnekleme" payı, sabit bir nicelik olarak **çürütüldü**: güven aralığı [0,00, 0,84] ve pay n'ye bağlı olarak 0,03'ten 0,25'e çıkıyor. API LLM'lerde pay 0,75–0,96.
- "İfade, örnekleme kadar büyük" iddiası, "benzer büyüklükte" anlamında **çürütüldü**: önceden kayıtlı hücrelerin 9/17'sinde ve API hücrelerinin 10/10'unda oran 3'ün üzerinde.
- Tabanın varlığı **doğrulandı** (API hücreleri dahil). 0,02 değeri ise araca özgü.
- n ≈ 2.000'de kesişme öngörüsü **çürütüldü**; kesişme n* = 3–302 aralığında.
- "Tanımsal belirsizlik" yorumu **zayıfladı**. Anlamı hiç değiştirmeyen seçenek sırası, LLM'lerde varyansın 0,11–0,19'unu alıyor; bu da yorumu daha da zayıflatıyor.
- "Kalibrasyon ifadesi değişince ortalama kapsama pek değişmez" iddiası **çürütüldü** (API: 0,755–0,897, en düşük 0,491).
- "Deterministik bir sınıflandırıcıda tekrarlanabilirlik tam sıfır olabilir" iddiası yerel CPU araçları için **doğrulandı**, API modelleri için **kısmen çürütüldü**: gpt-6-luna bit düzeyinde aynı, gpt-4.1-nano değil.
- Kirlenme (AG ve BBC karşılaştırması) LLM'ler için **açık**: iki API modeli σ_ifade'de zıt yönlerde.
- "Determinizm belirsizliği yok etmez, soruyu soranın seçimlerine taşır" iddiası **doğrulandı ve genişletildi**: API modellerinde belirsizliğin bir kısmı, kullanıcının seçmediği sunucu yığınında.

**Önceden kayıttan sapmalar ve bekleyen işler**
- NLI ailesinin sonuçları eksik (deberta BBC çalışıyor; diğer 8 T1 hücresi sırada).
- Yerel LLM sonuçları tek hücreden ibaret (Qwen BBC çalışıyor; SmolLM2 ve Qwen2.5-1.5B sırada). API hücreleri bunların yerine keşifsel olarak eklendi.
- Min-K% kirlenme ölçümü ile bi-encoder ve LLM hassasiyet koşulları henüz tamamlanmadı.

Ayrıntılar: `report.md` (§3.10 API-LLM'ler) ve `results/api/api_tables.md`.
