using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Examples;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public sealed class SupertonicTTSSample : MonoBehaviour
{
    // Language presets ordered as in the official Supertonic 3 web demo
    // (https://huggingface.co/spaces/Supertone/supertonic-3). Sample texts are
    // the "quote" presets from that demo's preset-texts.js.
    readonly struct LangPreset
    {
        public readonly string Code;
        public readonly string Label;
        public readonly string Sample;
        public LangPreset(string code, string label, string sample)
        {
            Code = code;
            Label = label;
            Sample = sample;
        }
    }

    static readonly LangPreset[] LangPresets =
    {
        new("en", "English (en)", "This text-to-speech system runs entirely in your browser, providing fast and private operation without sending any data to external servers."),
        new("ko", "한국어 (ko)", "이 텍스트 음성 변환 시스템은 브라우저에서 완전히 로컬로 동작하며, 모든 데이터를 기기 내에서 처리해 안전하고 빠른 성능을 제공합니다."),
        new("ja", "日本語 (ja)", "このテキスト読み上げシステムは、すべての処理がブラウザ内で完結します。データを外部のサーバーに送ることなく、高速かつプライベートに動作します。"),
        new("ar", "العربية (ar)", "يعمل نظام تحويل النص إلى كلام هذا بالكامل داخل متصفحك، ويوفر تشغيلاً سريعاً وخاصاً دون إرسال أي بيانات إلى خوادم خارجية."),
        new("bg", "Bulgarian (bg)", "Тази система за синтез на реч работи изцяло във вашия браузър, осигурявайки бърза и поверителна работа, без да изпраща никакви данни към външни сървъри."),
        new("cs", "Czech (cs)", "Tento systém pro převod textu na řeč běží zcela ve vašem prohlížeči, poskytuje rychlý a soukromý provoz a neodesílá žádná data na externí servery."),
        new("da", "Danish (da)", "Dette tekst-til-tale-system kører udelukkende i din browser og giver hurtig og privat drift uden at sende data til eksterne servere."),
        new("de", "Deutsch (de)", "Dieses Sprachsynthese-System läuft vollständig in Ihrem Browser, bietet schnelle und private Verarbeitung und sendet keine Daten an externe Server."),
        new("el", "Greek (el)", "Αυτό το σύστημα μετατροπής κειμένου σε ομιλία λειτουργεί εξ ολοκλήρου μέσα στον περιηγητή σας, παρέχοντας γρήγορη και ιδιωτική λειτουργία χωρίς να στέλνει δεδομένα σε εξωτερικούς διακομιστές."),
        new("es", "Español (es)", "Este sistema de conversión de texto a voz funciona completamente de forma local en el navegador y procesa todos los datos en el dispositivo para ofrecer un rendimiento rápido y seguro."),
        new("et", "Estonian (et)", "See teksti kõneks muutmise süsteem töötab täielikult teie veebilehitsejas, pakkudes kiiret ja privaatset talitlust ilma andmeid välistele serveritele saatmata."),
        new("fi", "Finnish (fi)", "Tämä tekstistä puheeksi -järjestelmä toimii kokonaan selaimessasi tarjoten nopean ja yksityisen toiminnan lähettämättä mitään tietoja ulkoisille palvelimille."),
        new("fr", "Français (fr)", "Ce système de synthèse vocale fonctionne entièrement en local dans le navigateur et traite toutes les données sur l'appareil afin d'offrir des performances rapides et sécurisées."),
        new("hi", "Hindi (hi)", "यह टेक्स्ट-टू-स्पीच प्रणाली पूरी तरह से आपके ब्राउज़र के भीतर ही चलती है। यह बाहरी सर्वर पर कोई डेटा भेजे बिना तेज़ और निजी संचालन प्रदान करती है।"),
        new("hr", "Croatian (hr)", "Ovaj sustav pretvaranja teksta u govor radi u potpunosti unutar vašeg preglednika, pružajući brz i privatan rad bez slanja ikakvih podataka vanjskim poslužiteljima."),
        new("hu", "Hungarian (hu)", "Ez a szövegfelolvasó rendszer teljes egészében a böngészőjében fut, gyors és bizalmas működést biztosítva anélkül, hogy bármilyen adatot küldene külső kiszolgálókra."),
        new("id", "Indonesian (id)", "Sistem pengubah teks menjadi ucapan ini berjalan sepenuhnya di dalam peramban Anda, memberikan operasi yang cepat dan pribadi tanpa mengirim data apa pun ke server eksternal."),
        new("it", "Italian (it)", "Questo sistema di sintesi vocale funziona interamente all'interno del tuo browser, offrendo un funzionamento rapido e privato senza inviare alcun dato a server esterni."),
        new("lt", "Lithuanian (lt)", "Ši teksto vertimo į kalbą sistema veikia visiškai jūsų naršyklėje, užtikrina greitą ir privatų darbą ir nesiunčia jokių duomenų į išorinius serverius."),
        new("lv", "Latvian (lv)", "Šī teksta pārveidošanas runā sistēma darbojas pilnībā jūsu pārlūkā un nodrošina ātru un privātu darbību, nesūtot nekādus datus uz ārējiem serveriem."),
        new("nl", "Dutch (nl)", "Dit tekst-naar-spraaksysteem werkt volledig in uw browser en biedt snelle en privé bediening zonder gegevens naar externe servers te sturen."),
        new("pl", "Polish (pl)", "Ten system zamiany tekstu na mowę działa w całości w przeglądarce, zapewniając szybkie i prywatne działanie bez wysyłania jakichkolwiek danych na zewnętrzne serwery."),
        new("pt", "Português (pt)", "Este sistema de conversão de texto em fala funciona totalmente de forma local no navegador, processando todos os dados no próprio dispositivo para garantir desempenho rápido e seguro."),
        new("ro", "Romanian (ro)", "Acest sistem de conversie a textului în vorbire rulează în întregime în browserul dumneavoastră, oferind o funcționare rapidă și privată, fără a trimite date către servere externe."),
        new("ru", "Russian (ru)", "Эта система синтеза речи работает полностью в вашем браузере, обеспечивая быструю и конфиденциальную работу и не отправляя никаких данных на внешние серверы."),
        new("sk", "Slovak (sk)", "Tento systém prevodu textu na reč beží úplne vo vašom prehliadači a poskytuje rýchlu a súkromnú prevádzku bez odosielania akýchkoľvek údajov na externé servery."),
        new("sl", "Slovenian (sl)", "Ta sistem za pretvorbo besedila v govor deluje v celoti v vašem brskalniku, zagotavlja hitro in zasebno delovanje brez pošiljanja kakršnih koli podatkov na zunanje strežnike."),
        new("sv", "Swedish (sv)", "Detta text-till-tal-system körs helt och hållet i din webbläsare och ger snabb och privat drift utan att skicka några data till externa servrar."),
        new("tr", "Turkish (tr)", "Bu metinden konuşmaya sistemi tamamen tarayıcınızın içinde çalışır, herhangi bir veriyi harici sunuculara göndermeden hızlı ve gizli işlem sağlar."),
        new("uk", "Ukrainian (uk)", "Ця система синтезу мовлення працює повністю у вашому браузері, забезпечуючи швидку та конфіденційну роботу без надсилання будь-яких даних на зовнішні сервери."),
        new("vi", "Vietnamese (vi)", "Hệ thống chuyển văn bản thành giọng nói này chạy hoàn toàn trong trình duyệt của bạn, mang lại hoạt động nhanh chóng và riêng tư mà không gửi bất kỳ dữ liệu nào đến máy chủ bên ngoài."),
    };

    [SerializeField] SupertonicTTS.Options[] platformOptions = { };

    [Header("UI References")]
    [SerializeField] TMP_InputField input;
    [SerializeField] TMP_Dropdown voiceDropdown;
    [SerializeField] TMP_Dropdown langDropdown;
    [SerializeField] Button generateButton;
    [SerializeField] TMP_Text statusLabel;
    [SerializeField] AudioSource audioSource;

    SupertonicTTS tts;

    async Awaitable Start()
    {
        RuntimeSystemFontFallback.Install();

        SetStatus("Loading model...");
        SetButtonEnabled(false);

        voiceDropdown.ClearOptions();
        voiceDropdown.AddOptions(SupertonicTTS.VoiceIds.ToList());

        langDropdown.ClearOptions();
        langDropdown.AddOptions(LangPresets.Select(p => p.Label).ToList());
        // Seed text from the currently-selected language before wiring the
        // change listener, so populating the dropdown doesn't fire it.
        input.text = LangPresets[langDropdown.value].Sample;
        langDropdown.onValueChanged.AddListener(OnLangChanged);

        var platform = Application.platform;
        var options = platformOptions.FirstOrDefault(o => o.platforms.Contains(platform));
        if (options == null || !options.TryGetModelPath(out var modelPath))
        {
            SetStatus("Model not found. Download it via:\n  hf download Supertone/supertonic-3 --local-dir <modelDir>");
            return;
        }

        try
        {
            tts = await SupertonicTTS.InitAsync(modelPath, destroyCancellationToken);
        }
        catch (OperationCanceledException)
        {
            return;
        }
        catch (Exception ex)
        {
            SetStatus($"Failed to load: {ex.Message}");
            Debug.LogException(ex);
            return;
        }

        SetStatus("Ready.");
        generateButton.onClick.AddListener(OnGenerateClick);
        SetButtonEnabled(true);
    }

    void OnDestroy()
    {
        if (generateButton != null)
        {
            generateButton.onClick.RemoveListener(OnGenerateClick);
        }
        if (langDropdown != null)
        {
            langDropdown.onValueChanged.RemoveListener(OnLangChanged);
        }
        if (audioSource != null && audioSource.clip != null)
        {
            Destroy(audioSource.clip);
        }
        tts?.Dispose();
    }

    void OnLangChanged(int index)
    {
        input.text = LangPresets[index].Sample;
    }

    async void OnGenerateClick()
    {
        string text = input.text;
        if (string.IsNullOrWhiteSpace(text)) return;

        string voiceId = SupertonicTTS.VoiceIds[voiceDropdown.value];
        string lang = LangPresets[langDropdown.value].Code;

        SetButtonEnabled(false);
        SetStatus($"Generating ({voiceId}, {lang})...");

        float[] pcm;
        try
        {
            pcm = await tts.GenerateAsync(text, voiceId, lang, destroyCancellationToken);
        }
        catch (OperationCanceledException)
        {
            return;
        }
        catch (Exception ex)
        {
            SetStatus($"Generation failed: {ex.Message}");
            Debug.LogException(ex);
            SetButtonEnabled(true);
            return;
        }

        if (audioSource.clip != null)
        {
            Destroy(audioSource.clip);
        }
        var clip = AudioClip.Create($"supertonic_{voiceId}", pcm.Length, 1, tts.SampleRate, false);
        clip.SetData(pcm, 0);
        audioSource.clip = clip;
        audioSource.Play();

        float seconds = (float)pcm.Length / tts.SampleRate;
        SetStatus($"Playing {seconds:F1}s ({voiceId}, {lang}).");
        SetButtonEnabled(true);
    }

    void SetStatus(string msg)
    {
        statusLabel.SetText(msg);
    }

    void SetButtonEnabled(bool enabled)
    {
        generateButton.interactable = enabled;
    }
}
