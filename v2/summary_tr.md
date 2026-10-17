# Özet (v2, ara rapor, 17 Ekim 2026)

Bu çalışma, sıfır-atışlı (zero-shot) metin sınıflandırmada konformal tahmin kümelerinin ortalama boyutu için GUM tarzı bir belirsizlik bütçesini yeniden sınıyor. v1'in iddiaları, `fab3c8a` commit'iyle önceden kaydedilen (preregistration) bir tasarımla test edildi. Tasarımın unsurları şunlar:
- geçerliliği iki bağımsız yargıç modelle denetlenmiş 30 rastgele etiket ifadesi seti,
- 200 rastgele kalibrasyon/test bölünmesi,
- iki yönlü rastgele etkiler modeli,
- beş veri seti (AG News, BBC 2024-25, SST-2, DBpedia-14, MASSIVE-18).

Sonuçlar ara niteliktedir: 16 sıfır-atışlı hücrenin 15'i bi-encoder, biri Qwen2.5-0.5B. NLI çapraz kodlayıcıların sonuçları henüz yok. Modeller yalnızca kendi türleri içinde karşılaştırıldı; üretken LLM'ler ile deterministik NLI ve bi-encoder sınıflandırıcılar birbirine karşı sıralanmadı.

**Ana bulgular**
- **Büyüklük.** n = 250'de etiket ifadesi bileşeni 16 hücrenin tamamında sıfırdan farklı. Bi-encoder'larda medyan σ_ifade/σ_örnekleme oranı 2,94, medyan ifade payı 0,75 (aralık 0,31–0,88). Qwen hücresinde pay 0,92.
- **Taban.** İfade bileşeni n ile küçülmüyor; eğim −0,04 ile 0,03 arasında. Örnekleme bileşeni ise −0,35 ile −0,87 arasındaki eğimlerle düşüyor. Taban, parafraz sayısından kaynaklanan bir yapay sonuç değil: W = 4 alt kümeleri, W = 30 tahmininin 0,87–0,96'sını veriyor.
- **Kapsama.** Aynı ifadeyle kalibre edildiğinde ifade, kapsamayı etkilemiyor (pay 0,00). Kalibrasyondan sonra ifade değişirse ortalama kapsama 0,732'ye kadar düşüyor (hedef 0,90).
- **Anlam mesafesi.** Bir etiketin anlamca uzaklığı tahmin kaymasını öngörüyor (bi-encoder ρ = 0,56). Qwen'de yalnızca büyük/küçük harf değişikliği, geçerli parafraz kaymasının 0,35'ini üretiyor.
- **Tekrarlanabilirlik (kısmi).** Aynı fp32 koşullarındaki iki bağımsız çalıştırmada max |Δp| = 0. Farklı batch bileşiminde fark 2e-6 ile 2e-5 arasında ve hiçbir tahmin değişmiyor.

**v1 iddialarına hükümler**
- "%32 ifade / %64 örnekleme" payı, sabit bir nicelik olarak **çürütüldü**: güven aralığı [0,00, 0,84] ve pay n'ye bağlı olarak 0,03'ten 0,25'e çıkıyor.
- "İfade, örnekleme kadar büyük" iddiası **doğrulandı**; çoğu hücrede ifade örneklemeden daha büyük.
- Tabanın varlığı **doğrulandı**. 0,02 değeri ise araca özgü.
- n ≈ 2.000'de kesişme öngörüsü **çürütüldü**; kesişme n* = 3–302 aralığında.
- "Tanımsal belirsizlik" yorumu **zayıfladı**: taban testi yalnızca sistematik ile rastgele etkiyi ayırıyor.
- "Kalibrasyon ifadesi değişince ortalama kapsama pek değişmez" iddiası **çürütüldü**.
- "Determinizm belirsizliği yok etmez, soruyu soranın seçimlerine taşır" iddiası **doğrulandı**.

**Önceden kayıttan sapmalar**
- NLI ailesinin sonuçları eksik.
- LLM sonuçları tek hücreden ibaret.
- Min-K% kirlenme ölçümü ve tekrarlanabilirlik koşulları henüz tamamlanmadı.

Ayrıntılar, sürmekte olan işler ve kaynaklar `report.md` dosyasında.
