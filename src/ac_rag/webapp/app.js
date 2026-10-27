const $ = (id) => document.getElementById(id);
const picker = $('document-picker');
const loaded = $('loaded-document');
const uploadStatus = $('upload-status');
const queryStatus = $('query-status');
const answerPanel = $('answer-panel');
const questionInput = $('question-input');
let currentDocument = '';
let busy = false;

function status(element, message, kind = '') {
  element.textContent = message;
  element.className = `inline-status ${kind}`;
}

function setBusy(value) {
  busy = value;
  $('sample-button').disabled = value;
  $('file-input').disabled = value;
  $('ask-button').disabled = value || !currentDocument;
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

function documentReady(name, isExample) {
  currentDocument = name;
  $('document-name').textContent = name;
  loaded.hidden = false;
  picker.hidden = true;
  $('suggestions').hidden = !isExample;
  answerPanel.hidden = true;
  status(queryStatus, '');
  setBusy(false);
  questionInput.focus();
}

async function useExample() {
  if (busy) return;
  setBusy(true);
  status(uploadStatus, 'Загружаем пример, при первом запуске это может занять минуту', 'busy');
  try {
    const response = await fetch('/example/load', {method: 'POST'});
    const data = await parseResponse(response);
    documentReady(data.source, true);
    status(uploadStatus, '');
  } catch (error) {
    status(uploadStatus, error.message, 'error');
    setBusy(false);
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
  status(uploadStatus, 'Загружаем файл, при первом запуске это может занять минуту', 'busy');
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch('/upload', {method: 'POST', body: form});
    const data = await parseResponse(response);
    documentReady(data.source, false);
    status(uploadStatus, '');
  } catch (error) {
    status(uploadStatus, error.message, 'error');
    setBusy(false);
  }
}

function showAnswer(data) {
  $('answer-text').textContent = data.answer;
  const source = data.sources?.[0];
  $('source-block').hidden = !source;
  if (source) {
    $('source-name').textContent = source.source;
    $('source-text').textContent = source.text;
  }
  $('answer-note').textContent = '';
  answerPanel.hidden = false;
  answerPanel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

async function ask(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !currentDocument || busy) return;
  setBusy(true);
  answerPanel.hidden = true;
  status(queryStatus, 'Ищем ответ', 'busy');
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
    setBusy(false);
  }
}

$('sample-button').addEventListener('click', useExample);
$('file-input').addEventListener('change', (event) => uploadFile(event.target.files[0]));
$('change-document').addEventListener('click', () => {
  currentDocument = '';
  loaded.hidden = true;
  picker.hidden = false;
  answerPanel.hidden = true;
  $('suggestions').hidden = true;
  $('file-input').value = '';
  questionInput.value = '';
  setBusy(false);
  status(queryStatus, '');
});
$('question-form').addEventListener('submit', ask);
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
