const $ = (id) => document.getElementById(id);
const picker = $('document-picker');
const loaded = $('loaded-document');
const uploadStatus = $('upload-status');
const uploadProgress = $('upload-progress');
const queryStatus = $('query-status');
const queryProgress = $('query-progress');
const answerPanel = $('answer-panel');
const comparison = $('comparison');
const questionInput = $('question-input');
const pdfDialog = $('pdf-dialog');
let currentDocument = '';
let busy = false;
let previewUrl = '';

function status(element, message, kind = '') {
  element.textContent = message;
  element.className = `inline-status ${kind}`;
}

function setBusy(value) {
  busy = value;
  $('sample-button').disabled = value;
  $('long-sample-button').disabled = value;
  $('file-input').disabled = value;
  $('ask-button').disabled = value || !currentDocument;
  $('compare-button').disabled = value || !currentDocument;
  questionInput.disabled = value || !currentDocument;
}

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) throw new Error('Для доступа к серверу нужен ключ');
    throw new Error(data.detail || 'Не удалось выполнить запрос, попробуйте ещё раз');
  }
  return data;
}

function documentReady(name, exampleKind = '') {
  currentDocument = name;
  $('document-name').textContent = name;
  loaded.hidden = false;
  picker.hidden = true;
  $('suggestions').hidden = exampleKind !== 'short';
  $('long-suggestions').hidden = exampleKind !== 'long';
  answerPanel.hidden = true;
  comparison.hidden = true;
  queryProgress.hidden = true;
  status(queryStatus, '');
  setBusy(false);
  questionInput.focus();
}

async function useExample() {
  if (busy) return;
  setBusy(true);
  status(uploadStatus, 'Подготавливаем пример, при первом запуске это может занять минуту', 'busy');
  uploadProgress.hidden = false;
  try {
    const response = await fetch('/example/load', {method: 'POST'});
    const data = await parseResponse(response);
    documentReady(data.source, 'short');
    status(uploadStatus, '');
  } catch (error) {
    status(uploadStatus, error.message, 'error');
    setBusy(false);
  } finally {
    uploadProgress.hidden = true;
  }
}

async function useLongExample() {
  if (busy) return;
  setBusy(true);
  status(uploadStatus, 'Индексируем длинный пример, это может занять несколько минут', 'busy');
  uploadProgress.hidden = false;
  try {
    const response = await fetch('/example/long/load', {method: 'POST'});
    const data = await parseResponse(response);
    documentReady(data.source, 'long');
    status(uploadStatus, '');
  } catch (error) {
    status(uploadStatus, error.message, 'error');
    setBusy(false);
  } finally {
    uploadProgress.hidden = true;
  }
}

async function uploadFile(file) {
  if (!file || busy) return;
  if (!/\.(txt|md|pdf|docx)$/i.test(file.name)) {
    status(uploadStatus, 'Выберите TXT, MD, PDF или DOCX', 'error');
    return;
  }
  if (file.size > 5_000_000) {
    status(uploadStatus, 'Файл должен быть не больше 5 МБ', 'error');
    return;
  }
  setBusy(true);
  status(uploadStatus, 'Загружаем и обрабатываем файл, при первом запуске это может занять минуту', 'busy');
  uploadProgress.hidden = false;
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch('/upload', {method: 'POST', body: form});
    const data = await parseResponse(response);
    documentReady(data.source);
    status(uploadStatus, '');
  } catch (error) {
    status(uploadStatus, error.message, 'error');
    setBusy(false);
  } finally {
    uploadProgress.hidden = true;
  }
}

function showAnswer(data) {
  comparison.hidden = true;
  $('answer-text').textContent = data.answer;
  const sources = data.sources || [];
  const sourceById = new Map(sources.map((source) => [source.id, source]));
  const sourceItemById = new Map();
  $('source-block').hidden = !sources.length;
  const sourceList = $('source-list');
  sourceList.replaceChildren();
  for (const source of sources) {
    const item = document.createElement('div');
    item.className = 'source-item';
    const name = document.createElement('div');
    name.className = 'source-name';
    name.textContent = source.page ? `${source.source} · стр. ${source.page}` : source.source;
    const text = document.createElement('p');
    text.className = 'source-text';
    text.textContent = source.text;
    item.append(name, text);
    if (source.page && source.source.toLowerCase().endsWith('.pdf')) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'text-button source-open';
      button.textContent = `Открыть страницу ${source.page}`;
      button.addEventListener('click', () => openPdfSource(source));
      item.append(button);
    }
    sourceList.append(item);
    sourceItemById.set(source.id, item);
  }
  const evidenceList = $('evidence-list');
  evidenceList.replaceChildren();
  for (const entry of data.evidence || []) {
    const row = document.createElement('div');
    row.className = 'evidence-row';
    const sentence = document.createElement('span');
    sentence.textContent = entry.sentence;
    row.append(sentence);
    for (const id of entry.citations) {
      const source = sourceById.get(id);
      if (!source) continue;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'evidence-link';
      button.textContent = source.page ? `Стр. ${source.page}` : 'Фрагмент';
      button.addEventListener('click', () => {
        if (source.page && source.source.toLowerCase().endsWith('.pdf')) openPdfSource(source);
        else sourceItemById.get(id)?.scrollIntoView({behavior: 'smooth', block: 'center'});
      });
      row.append(button);
    }
    evidenceList.append(row);
  }
  $('evidence-block').hidden = !evidenceList.childElementCount;
  $('answer-note').textContent = '';
  answerPanel.hidden = false;
  answerPanel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function showComparison(data) {
  answerPanel.hidden = true;
  for (const kind of ['baseline', 'adaptive']) {
    const result = data[kind];
    $(`${kind}-answer`).textContent = result.answer;
    const tokens = result.input_tokens == null ? '' : ` · ${result.input_tokens + result.output_tokens} токенов`;
    $(`${kind}-metrics`).textContent = `${(result.latency_ms / 1000).toFixed(1)} с · ${result.llm_calls} вызовов модели · ${result.citations.length} источников${tokens}`;
    const source = result.sources?.[0];
    const sourceElement = $(`${kind}-source`);
    sourceElement.textContent = source ? `Источник: ${source.source} — ${source.text}` : 'Подтверждающий фрагмент не указан';
    if (source?.page && source.source.toLowerCase().endsWith('.pdf')) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'text-button source-open';
      button.textContent = `Открыть страницу ${source.page}`;
      button.addEventListener('click', () => openPdfSource(source));
      sourceElement.append(button);
    }
  }
  comparison.hidden = false;
  comparison.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

async function openPdfSource(source) {
  $('pdf-title').textContent = `${source.source} · стр. ${source.page}`;
  $('pdf-status').textContent = 'Открываем страницу';
  $('pdf-preview').hidden = true;
  $('pdf-original').href = `/source-pdf?source=${encodeURIComponent(source.source)}#page=${source.page}`;
  pdfDialog.showModal();
  try {
    const url = `/source-preview?source=${encodeURIComponent(source.source)}&chunk_id=${source.id}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Не удалось открыть страницу. Загрузите PDF снова');
    const blob = await response.blob();
    if (!pdfDialog.open) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(blob);
    $('pdf-preview').src = previewUrl;
    $('pdf-preview').hidden = false;
    $('pdf-status').textContent = response.headers.get('X-Highlighted') === 'true' ? 'Фрагмент источника выделен зелёным' : 'Страница найдена, но фрагмент не удалось выделить точно';
  } catch (error) {
    $('pdf-status').textContent = error.message;
  }
}

$('pdf-close').addEventListener('click', () => pdfDialog.close());
pdfDialog.addEventListener('close', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = '';
  $('pdf-preview').removeAttribute('src');
});

async function ask(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !currentDocument || busy) return;
  setBusy(true);
  answerPanel.hidden = true;
  status(queryStatus, 'Ищем ответ', 'busy');
  queryProgress.hidden = false;
  try {
    const response = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: question, document_only: true, source: currentDocument}),
    });
    showAnswer(await parseResponse(response));
    status(queryStatus, '');
  } catch (error) {
    status(queryStatus, error.message, 'error');
  } finally {
    queryProgress.hidden = true;
    setBusy(false);
  }
}

async function compareMethods() {
  const question = questionInput.value.trim();
  if (!question || !currentDocument || busy) return;
  setBusy(true);
  answerPanel.hidden = true;
  comparison.hidden = true;
  status(queryStatus, 'Сравниваем методы', 'busy');
  queryProgress.hidden = false;
  try {
    const response = await fetch('/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: question, document_only: true, source: currentDocument}),
    });
    showComparison(await parseResponse(response));
    status(queryStatus, '');
  } catch (error) {
    status(queryStatus, error.message, 'error');
  } finally {
    queryProgress.hidden = true;
    setBusy(false);
  }
}

$('sample-button').addEventListener('click', useExample);
$('long-sample-button').addEventListener('click', useLongExample);
$('file-input').addEventListener('change', (event) => uploadFile(event.target.files[0]));
$('change-document').addEventListener('click', () => {
  currentDocument = '';
  loaded.hidden = true;
  picker.hidden = false;
  answerPanel.hidden = true;
  comparison.hidden = true;
  queryProgress.hidden = true;
  uploadProgress.hidden = true;
  $('suggestions').hidden = true;
  $('long-suggestions').hidden = true;
  $('file-input').value = '';
  questionInput.value = '';
  setBusy(false);
  status(queryStatus, '');
});
$('question-form').addEventListener('submit', ask);
$('compare-button').addEventListener('click', compareMethods);
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => {
  questionInput.value = button.dataset.question;
  questionInput.focus();
}));
const dropZone = $('drop-zone');
dropZone.addEventListener('dragover', (event) => {event.preventDefault(); dropZone.classList.add('dragover');});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragover');
  uploadFile(event.dataTransfer.files[0]);
});
