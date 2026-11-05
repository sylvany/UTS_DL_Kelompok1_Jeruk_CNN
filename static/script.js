const uploadArea   = document.getElementById('uploadArea');
const inputGambar  = document.getElementById('inputGambar');
const previewBox   = document.getElementById('previewBox');
const previewImg   = document.getElementById('previewImg');
const namaFile     = document.getElementById('namaFile');
const btnProses    = document.getElementById('btnProses');
const loading      = document.getElementById('loading');
const hasilSection = document.getElementById('hasilSection');
const stepNavBottom = document.getElementById('stepNavBottom');
const stepCounter  = document.getElementById('stepCounter');
const btnPrev      = document.getElementById('btnPrev');
const btnNext      = document.getElementById('btnNext');

let selectedFile = null;
const steps = ['m1','m2','m3','m4','m5','m6'];
let currentStep = 0;

function goToStep(idx) {
    currentStep = idx;
    steps.forEach((id, i) => {
        const btn = document.querySelector(`.step-item[data-menu="${id}"]`);
        const panel = document.getElementById('panel-' + id);
        if (!btn || !panel) return;
        btn.classList.toggle('active', i === idx);
        btn.classList.toggle('done', i < idx);
        panel.classList.toggle('active', i === idx);
    });
    stepCounter.textContent = `Step ${idx + 1} / ${steps.length}`;
    btnPrev.disabled = idx === 0;
    btnNext.disabled = idx === steps.length - 1;
    btnNext.classList.toggle('primary', idx < steps.length - 1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.querySelectorAll('.step-item').forEach((btn, i) => {
    btn.addEventListener('click', () => {
        if (hasilSection.style.display !== 'none') goToStep(i);
    });
});

btnPrev.addEventListener('click', () => { if (currentStep > 0) goToStep(currentStep - 1); });
btnNext.addEventListener('click', () => { if (currentStep < steps.length - 1) goToStep(currentStep + 1); });

uploadArea.addEventListener('click', () => inputGambar.click());

uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) tampilkanPreview(file);
});

inputGambar.addEventListener('change', () => {
    if (inputGambar.files[0]) tampilkanPreview(inputGambar.files[0]);
});

function tampilkanPreview(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        namaFile.innerHTML = `<span>${file.name}</span><button id="btnGanti" onclick="document.getElementById('inputGambar').click()">🔄 Ganti Gambar</button>`;
        previewBox.style.display = 'block';
        uploadArea.style.display = 'none';
        btnProses.disabled = false;
        hasilSection.style.display = 'none';
        stepNavBottom.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

const uploadAreaQuick   = document.getElementById('uploadAreaQuick');
const inputGambarQuick  = document.getElementById('inputGambarQuick');
const previewBoxQuick   = document.getElementById('previewBoxQuick');
const previewImgQuick   = document.getElementById('previewImgQuick');
const namaFileQuick     = document.getElementById('namaFileQuick');
const btnProsesQuick    = document.getElementById('btnProsesQuick');
const quickResultModal  = document.getElementById('quickResultModal');

let selectedFileQuick = null;

uploadAreaQuick.addEventListener('click', () => inputGambarQuick.click());

uploadAreaQuick.addEventListener('dragover', (e) => { e.preventDefault(); uploadAreaQuick.classList.add('drag-over'); });
uploadAreaQuick.addEventListener('dragleave', () => uploadAreaQuick.classList.remove('drag-over'));
uploadAreaQuick.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadAreaQuick.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) tampilkanPreviewQuick(file);
});

inputGambarQuick.addEventListener('change', () => {
    if (inputGambarQuick.files[0]) tampilkanPreviewQuick(inputGambarQuick.files[0]);
});

function tampilkanPreviewQuick(file) {
    selectedFileQuick = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImgQuick.src = e.target.result;
        namaFileQuick.innerHTML = `<span>${file.name}</span><button onclick="document.getElementById('inputGambarQuick').click()">🔄 Ganti</button>`;
        previewBoxQuick.style.display = 'block';
        uploadAreaQuick.style.display = 'none';
        btnProsesQuick.disabled = false;
    };
    reader.readAsDataURL(file);
}

btnProsesQuick.addEventListener('click', async () => {
    if (!selectedFileQuick) return;
    btnProsesQuick.disabled = true;

    const formData = new FormData();
    formData.append('gambar', selectedFileQuick);

    try {
        const res = await fetch('/deteksi-cepat', { method: 'POST', body: formData });
        const data = await res.json();

        document.getElementById('modalQuickImg').src = data.original;
        
        const isFresh = data.status === 'normal';
        const isRagu  = data.status === 'tidak_yakin';
        const badge = document.getElementById('modalQuickBadge');
        badge.textContent = isRagu ? '❓ TIDAK YAKIN' : (isFresh ? '✅ SEGAR' : '⚠️ RUSAK');
        badge.style.background = isRagu ? '#FFFBEB' : (isFresh ? '#ECFDF5' : '#FEF2F2');
        badge.style.color = isRagu ? '#92400E' : (isFresh ? '#065F46' : '#991B1B');
        badge.style.border = isRagu ? '1px solid #FDE68A' : (isFresh ? '1px solid #A7F3D0' : '1px solid #FECACA');

        document.getElementById('modalQuickStatus').textContent = isRagu ? '❓ Model Tidak Yakin' : (isFresh ? '✅ Buah Jeruk SEGAR' : '⚠️ Buah Jeruk RUSAK');
        tampilkanAmbang('modalQuickAmbang', data);
        document.getElementById('modalQuickCategory').innerHTML = `<strong>${data.kategori}</strong> — Confidence: ${data.confidence}%`;

        const confBar = document.getElementById('modalQuickConfBar');
        confBar.style.width = data.confidence + '%';
        confBar.style.background = isRagu ? '#D97706' : (isFresh ? '#16A34A' : '#DC2626');
        document.getElementById('modalQuickConfPct').textContent = data.confidence + '%';

        const probaEl = document.getElementById('modalQuickProba');
        probaEl.innerHTML = '';
        ['blackspot','canker','fresh','grenning'].forEach((key, i) => {
            const pct = Math.round((data.proba_cnn[i] || 0) * 1000) / 10;
            const row = document.createElement('div');
            row.className = 'modal-proba-row';
            row.innerHTML = `
                <div class="modal-proba-label">${catLabels[key]}</div>
                <div class="modal-proba-track"><div class="modal-proba-fill" style="width:${pct}%;background:${catColors[key]}"></div></div>
                <div class="modal-proba-pct">${pct}%</div>`;
            probaEl.appendChild(row);
        });

        window.lastQuickDetectionData = { ...data, file: selectedFileQuick };
        window.lastQuickDetectionImg = data.original;

        quickResultModal.style.display = 'flex';

    } catch (err) {
        alert('Terjadi kesalahan saat deteksi. Coba lagi.');
        console.error(err);
    } finally {
        btnProsesQuick.disabled = false;
    }
});

function closeQuickModal() {
    quickResultModal.style.display = 'none';
    previewBoxQuick.style.display = 'none';
    uploadAreaQuick.style.display = 'block';
    inputGambarQuick.value = '';
    selectedFileQuick = null;
    btnProsesQuick.disabled = true;
}

function showFullAnalysis() {
    if (!window.lastQuickDetectionData) return;
    closeQuickModal();
    tampilkanPreview(window.lastQuickDetectionData.file);
    setTimeout(() => btnProses.click(), 100);
}

const catColors = { blackspot:'#374151', canker:'#DC2626', fresh:'#16A34A', grenning:'#CA8A04' };
const catLabels = { blackspot:'Blackspot', canker:'Canker', fresh:'Fresh', grenning:'Grenning' };

function tampilkanAmbang(elId, data) {
    const el = document.getElementById(elId);
    if (data.diterima === false) {
        el.style.display = 'block';
        el.innerHTML = `<strong>❓ Model tidak yakin.</strong> Keyakinan tertinggi ${data.confidence}% berada di bawah ambang ${data.ambang}%.
            Dugaan terdekat: ${data.kategori}, belum dapat dipercaya. Coba unggah foto jeruk yang lebih jelas.`;
    } else {
        el.style.display = 'none';
        el.innerHTML = '';
    }
}

function buildProbaBars(containerId, probaArr) {
    const keys = ['blackspot','canker','fresh','grenning'];
    const el = document.getElementById(containerId);
    el.innerHTML = '';
    keys.forEach((key, i) => {
        const pct = Math.round((probaArr[i] || 0) * 1000) / 10;
        const row = document.createElement('div');
        row.className = 'proba-row';
        row.innerHTML = `
            <div class="proba-label">${catLabels[key]}</div>
            <div class="proba-track"><div class="proba-fill" style="width:${pct}%;background:${catColors[key]}"></div></div>
            <div class="proba-pct">${pct}%</div>`;
        el.appendChild(row);
    });
}

btnProses.addEventListener('click', async () => {
    if (!selectedFile) return;
    loading.style.display      = 'flex';
    hasilSection.style.display = 'none';
    stepNavBottom.style.display = 'none';
    btnProses.disabled         = true;

    const formData = new FormData();
    formData.append('gambar', selectedFile);

    try {
        const res  = await fetch('/proses', { method: 'POST', body: formData });
        const data = await res.json();

        document.getElementById('imgOriginal').src  = data.original;
        document.getElementById('infoUkuran').textContent  = data.info.ukuran;
        document.getElementById('infoChannel').textContent = data.info.channel;
        document.getElementById('infoPixel').textContent   = data.info.total_piksel;
        document.getElementById('infoFormat').textContent  = data.info.format;
        const predBox = document.getElementById('prediksiAwal');
        predBox.textContent  = (data.status === 'tidak_yakin' ? 'Tidak yakin: ' : '') + data.kategori + ' (' + data.confidence + '%)';
        predBox.className    = 'prediksi-box ' + (data.status === 'tidak_yakin' ? 'prediksi-ragu' : (data.status === 'rusak' ? 'prediksi-rusak' : 'prediksi-normal'));

        document.getElementById('imgDenoised2').src = data.original;
        document.getElementById('imgNoisy').src     = data.m2_noisy;
        document.getElementById('imgDenoised').src  = data.m2_denoised;
        document.getElementById('imgResize').src    = data.m2_resize;

        document.getElementById('imgResize2').src = data.m2_resize;
        document.getElementById('imgBlur').src    = data.m3_blur;
        document.getElementById('imgSharp').src   = data.m3_sharp;

        document.getElementById('imgHsv').src       = data.m4_hsv;
        document.getElementById('imgMask').src      = data.m4_mask;
        document.getElementById('imgDilasi').src    = data.m4_dilasi;
        document.getElementById('imgErosi').src     = data.m4_erosi;
        document.getElementById('imgSegmented').src = data.m4_segmented;

        document.getElementById('imgEdges').src       = data.m5_edges;
        document.getElementById('imgEdgeOverlay').src = data.m5_overlay;
        document.getElementById('imgHough').src       = data.m5_hough;
        document.getElementById('imgResult').src      = data.m5_result;

        document.getElementById('cnnResult').textContent = data.cat_cnn;
        document.getElementById('cnnConf').textContent   = 'Confidence: ' + data.conf_cnn + '%';
        buildProbaBars('cnnBars', data.proba_cnn);

        const k = document.getElementById('kesimpulan');
        if (data.status === 'tidak_yakin') {
            k.className = 'kesimpulan-box kesimpulan-ragu';
            k.innerHTML = `
                <div class="kesimpulan-judul">❓ Model Tidak Yakin</div>
                <div class="kesimpulan-persen">Keyakinan tertinggi: <strong>${data.confidence}%</strong> (ambang ${data.ambang}%)</div>
                <div class="kesimpulan-persen">Dugaan terdekat: <strong>${data.kategori}</strong>, belum dapat dipercaya</div>
                <hr style="margin:0.5rem 0;border-color:#FDE68A">
                <div class="kesimpulan-persen">🧠 CNN: ${data.cat_cnn} (${data.conf_cnn}%)</div>`;
        } else if (data.status === 'rusak') {
            k.className = 'kesimpulan-box kesimpulan-rusak';
            k.innerHTML = `
                <div class="kesimpulan-judul">⚠️ Buah Jeruk Terdeteksi RUSAK</div>
                <div class="kesimpulan-persen">Kategori: <strong>${data.kategori}</strong></div>
                <div class="kesimpulan-persen">Model: <strong>${data.model_used}</strong></div>
                <div class="kesimpulan-persen">Keyakinan: <strong>${data.confidence}%</strong></div>
                <hr style="margin:0.5rem 0;border-color:#FECACA">
                <div class="kesimpulan-persen">🧠 CNN: ${data.cat_cnn} (${data.conf_cnn}%)</div>
                <div class="kesimpulan-persen">📦 Bounding Box: <strong>${data.bbox_count}</strong> area &nbsp;|&nbsp; Kerusakan: <strong>${data.damage_pct}%</strong></div>`;
        } else {
            k.className = 'kesimpulan-box kesimpulan-normal';
            k.innerHTML = `
                <div class="kesimpulan-judul">✅ Buah Jeruk Terdeteksi NORMAL / SEGAR</div>
                <div class="kesimpulan-persen">Kategori: <strong>${data.kategori}</strong></div>
                <div class="kesimpulan-persen">Model: <strong>${data.model_used}</strong></div>
                <div class="kesimpulan-persen">Keyakinan: <strong>${data.confidence}%</strong></div>
                <hr style="margin:0.5rem 0;border-color:#A7F3D0">
                <div class="kesimpulan-persen">🧠 CNN: ${data.cat_cnn} (${data.conf_cnn}%)</div>`;
        }

        document.getElementById('imgOriginal2').src = data.original;
        document.getElementById('imgResult2').src   = data.m5_result;
        document.getElementById('imgGradcam').src = data.gradcam;
        tampilkanAmbang('ambangBox', data);
        buildProbaBars('m6Bars', data.proba_cnn);
        document.getElementById('m6Result').innerHTML =
            `Kategori: <strong>${data.m6_kategori}</strong> &nbsp;|&nbsp; Confidence (CNN): <strong>${data.m6_conf}%</strong>`;

        hasilSection.style.display  = 'block';
        stepNavBottom.style.display = 'flex';
        goToStep(0);
        hasilSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        alert('Terjadi kesalahan saat memproses gambar. Coba lagi.');
        console.error(err);
    } finally {
        loading.style.display = 'none';
        btnProses.disabled    = false;
    }
});