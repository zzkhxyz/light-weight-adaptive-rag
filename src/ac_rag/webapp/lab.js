const slider = document.getElementById('length-slider');
const lengthLabel = document.getElementById('length-label');
const results = document.getElementById('lab-results');
const labStatus = document.getElementById('lab-status');
let lengths = [];
let cases = [];

function tokenLabel(count) {
  const lastTwo = count % 100;
  const last = count % 10;
  const ending = lastTwo >= 11 && lastTwo <= 14 ? 'ов' : last === 1 ? '' : last >= 2 && last <= 4 ? 'а' : 'ов';
  return `${count} токен${ending}`;
}

function render() {
  const selected = lengths[Number(slider.value)];
  lengthLabel.textContent = tokenLabel(selected);
  results.replaceChildren();
  for (const position of ['start', 'middle', 'end']) {
    const item = cases.find((row) => row.tokens === selected && row.position === position);
    const card = document.createElement('article');
    card.className = 'comparison-card';
    const title = document.createElement('h3');
    title.textContent = {start: 'Факт в начале', middle: 'Факт в середине', end: 'Факт в конце'}[position];
    const answer = document.createElement('p');
    answer.className = 'comparison-answer';
    answer.textContent = item.answer || 'Нет ответа';
    const metrics = document.createElement('div');
    metrics.className = 'comparison-metrics';
    metrics.textContent = `${item.recalled ? 'Факт найден' : 'Факт не найден'} · состояние ${(item.state_bytes / 1048576).toFixed(2)} МиБ · ${(item.latency_ms / 1000).toFixed(1)} с`;
    card.append(title, answer, metrics);
    results.append(card);
  }
}

fetch('/assets/ssm-report.json').then((response) => {
  if (!response.ok) throw new Error('Результаты ещё не подготовлены');
  return response.json();
}).then((report) => {
  cases = report.cases;
  lengths = [...new Set(cases.map((row) => row.tokens))].sort((a, b) => a - b);
  slider.max = String(lengths.length - 1);
  slider.disabled = false;
  slider.addEventListener('input', render);
  render();
}).catch((error) => { labStatus.textContent = error.message; });
