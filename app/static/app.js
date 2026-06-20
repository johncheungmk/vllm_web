let profiles = [];
let current = null;
let vllmOptions = {choices: {}, supported_cli_options: [], detected: false, warnings: []};
let dirty = false;
const $ = (id) => document.getElementById(id);
const form = $('profile-form');
const checkboxFields = new Set([
  'trust_remote_code','disable_log_stats','enable_prefix_caching','enable_chunked_prefill','enforce_eager',
  'data_parallel_hybrid_lb','data_parallel_external_lb','data_parallel_multi_port_external_lb','headless',
  'speculative_enabled','is_moe_model','external_load_balancer','enable_expert_parallel','enable_ep_weight_filter'
]);
const intFields = new Set(['port','max_model_len','max_num_seqs','max_num_batched_tokens','tensor_parallel_size','pipeline_parallel_size','data_parallel_size','data_parallel_size_local','data_parallel_start_rank','data_parallel_rpc_port','data_parallel_rank','api_server_count','ray_port','ray_num_nodes','ray_gpus_per_node','ray_min_nodes','spark_executor_instances','spark_executor_cores','spark_executor_gpus','num_speculative_tokens','draft_tensor_parallel_size','prompt_lookup_min','prompt_lookup_max']);
const floatFields = new Set(['gpu_memory_utilization','swap_space','cpu_offload_gb']);
const jsonFields = new Set([
  'env_json',
  'speculative_extra_json',
  'spark_conf_json',
  'structured_outputs_config',
  'compilation_config',
  'kv_transfer_config',
  'additional_config',
  'override_generation_config'
]);
const requiredFields = new Map([
  ['name', 'Profile name'],
  ['model', 'Model'],
  ['host', 'Host'],
  ['port', 'Port'],
  ['tensor_parallel_size', 'Tensor parallel size'],
  ['pipeline_parallel_size', 'Pipeline parallel size'],
  ['ray_port', 'Ray port']
]);
let previewTimer = null;
const wizardSteps = ['basic', 'performance', 'parallel', 'speculative', 'review'];
let currentWizardIndex = 0;
const tooltipText = {
  'profile-select': 'Choose which saved vLLM profile to edit or launch.',
  'new-profile': 'Create a new profile from the current settings.',
  'clone-profile': 'Duplicate the selected profile.',
  'delete-profile': 'Delete the selected profile. Running profiles must be stopped first.',
  'save-profile': 'Save the current form values to the selected profile.',
  'start-server': 'Start vLLM serve with the current form values.',
  'restart-server': 'Restart vLLM using the saved selected profile.',
  'stop-server': 'Stop the vLLM process started by vLLM Web.',
  'export-startup-script': 'Generate a shell script for manually starting this vLLM profile.',
  'export-systemd': 'Generate an example systemd service for this profile.',
  'export-ray': 'Generate a Ray cluster launch script for this profile.',
  'export-spark': 'Generate a Spark-managed Ray launch script. Expert Spark/Ray deployments only.',
  'export-profile': 'Export only the selected profile as JSON.',
  'export-all-profiles': 'Export every saved profile as JSON.',
  'import-profiles': 'Import profile JSON into vLLM Web.',
  'copy-command': 'Copy the current command preview.',
  'copy-wizard-command': 'Copy the wizard command preview.',
  'wizard-download-script': 'Download a start-vllm.sh script for the current wizard settings.',
  'wizard-start-server': 'Start vLLM directly from vLLM Web using the current wizard settings.',
  'send-test': 'Send a chat completion request to the active vLLM server.',
  'refresh-gpu': 'Refresh GPU telemetry from nvidia-smi when available.'
};
const fieldHelp = {
  name: 'Human-readable name for this launch profile.',
  model: 'Model path or Hugging Face model ID passed to vllm serve.',
  served_model_name: 'Optional model alias exposed by the OpenAI-compatible API.',
  host: 'Host interface for the vLLM API server. Use 127.0.0.1 for local-only access.',
  port: 'TCP port for the vLLM API server.',
  api_key: 'Optional bearer token required by clients calling the vLLM API.',
  dtype: 'Model weight dtype. auto lets vLLM choose a compatible default.',
  quantization: 'Quantization backend if the model or deployment needs one.',
  generation_config: 'Generation config source used by vLLM.',
  gpu_memory_utilization: 'Fraction of GPU memory vLLM can reserve for KV cache.',
  max_model_len: 'Maximum context length vLLM should allow for this model.',
  max_num_seqs: 'Maximum concurrent sequences. Higher values can increase throughput and memory use.',
  max_num_batched_tokens: 'Maximum batched tokens. Higher values can improve throughput but use more memory.',
  kv_cache_dtype: 'KV cache dtype. Lower precision can reduce memory but may affect compatibility.',
  performance_mode: 'Preset performance policy when supported by your vLLM version.',
  tensor_parallel_size: 'Number of GPUs used for tensor parallelism.',
  pipeline_parallel_size: 'Number of pipeline stages, often matching node count for multi-node Ray.',
  distributed_executor_backend: 'Execution backend. Use mp for single-node multi-GPU or ray for multi-node.',
  ray_head_address: 'Ray head node IP or hostname for multi-node deployments.',
  ray_port: 'Ray head node port.',
  ray_num_nodes: 'Number of Ray nodes in the deployment.',
  ray_gpus_per_node: 'GPU count per Ray node.',
  ray_node_ips: 'Node IP list used for exported Ray runbooks and scripts.',
  speculative_method: 'Speculative decoding method. Leave none unless you know the target model supports it.',
  speculative_model: 'Draft model, EAGLE head, MTP checkpoint, or custom proposer class path.',
  num_speculative_tokens: 'Number of speculative tokens proposed per step.',
  cuda_visible_devices: 'Optional CUDA_VISIBLE_DEVICES value for the vLLM process.',
  env_json: 'Extra environment variables as a JSON object.',
  advanced_args: 'Additional vLLM CLI arguments parsed safely with shlex.'
};

async function api(path, options = {}) {
  const res = await fetch(path, {headers: {'Content-Type':'application/json'}, ...options});
  const text = await res.text();
  let body;
  try { body = text ? JSON.parse(text) : {}; } catch { body = text; }
  if (!res.ok) {
    const detail = body && body.detail;
    const message = friendlyError(detail || body || text || res.statusText);
    throw new Error(message);
  }
  return body;
}

function activeProfileId() { return $('profile-select').value; }
function profileById(id) { return profiles.find(p => p.id === id); }
function markDirty(isDirty = true) {
  dirty = isDirty;
  $('dirty-badge').hidden = !dirty;
}

async function loadVllmOptions() {
  try {
    vllmOptions = await api('/api/vllm/options');
    const label = vllmOptions.detected ? `vLLM detected: ${escapeHtml(vllmOptions.version || 'installed')}` : 'vLLM not detected; using fallback option list.';
    const firstWarning = (vllmOptions.warnings || []).find(w => !w.startsWith(label));
    $('vllm-info').innerHTML = `${label}${firstWarning ? `<br>${escapeHtml(firstWarning)}` : ''}`;
    populateOptionSelects();
    applyUnsupportedVisibility();
  } catch (e) {
    $('vllm-info').textContent = `Could not load vLLM options: ${e.message}`;
  }
}

function populateOptionSelects() {
  document.querySelectorAll('select[data-option-key]').forEach(select => {
    const key = select.dataset.optionKey;
    const choices = vllmOptions.choices?.[key];
    if (!choices || !choices.length) return;
    const existingCustom = Array.from(select.options).some(option => option.value === '__custom__');
    const currentValue = select.value;
    const existingBlank = select.querySelector('option[value=""]');
    const blankLabel = existingBlank?.textContent || (key === 'quantization' ? 'model default / none' : 'default');
    select.innerHTML = '';
    if (choices.includes('') || existingBlank) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = blankLabel;
      select.appendChild(option);
    }
    choices.filter(value => value !== '').forEach(value => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
    if (existingCustom) {
      const custom = document.createElement('option');
      custom.value = '__custom__';
      custom.textContent = 'custom...';
      select.appendChild(custom);
    }
    if (Array.from(select.options).some(option => option.value === currentValue)) select.value = currentValue;
  });
}

function applyUnsupportedVisibility() {
  const supported = new Set(vllmOptions.supported_cli_options || []);
  const detected = Boolean(vllmOptions.detected);
  document.querySelectorAll('[data-flag]').forEach(el => {
    const flag = el.dataset.flag;
    el.hidden = detected && !supported.has(flag);
  });
}

function setupTooltips() {
  document.querySelectorAll('.field-help').forEach(help => {
    const label = help.closest('label');
    if (label && !label.dataset.help) label.dataset.help = help.textContent.trim();
  });
  Object.entries(fieldHelp).forEach(([name, text]) => {
    const el = form.elements[name];
    const label = el?.closest?.('label');
    if (label && !label.dataset.help) label.dataset.help = text;
  });
  Object.entries(tooltipText).forEach(([id, text]) => {
    const el = $(id);
    if (el && !el.dataset.help) el.dataset.help = text;
  });
}

function friendlyError(detail) {
  if (typeof detail === 'string') return detail;
  if (detail?.errors || detail?.warnings) return [...(detail.errors || []), ...(detail.warnings || [])].join('\n');
  if (Array.isArray(detail)) {
    return detail.map(item => {
      const loc = Array.isArray(item.loc) ? item.loc.filter(part => part !== 'body').join('.') : '';
      return `${loc ? `${loc}: ` : ''}${item.msg || JSON.stringify(item)}`;
    }).join('\n');
  }
  return String(detail);
}

function setMessage(message, type = 'error') {
  showValidation({[type === 'error' ? 'errors' : 'warnings']: [message]});
}
function showValidation(report = {}) {
  const box = $('validation-box');
  if (!box) return;
  const warnings = [...(report.warnings || [])];
  const messages = [...(report.messages || [])];
  if (report.dashboard?.warning) warnings.push(report.dashboard.warning);
  const errors = report.errors || [];
  if (!warnings.length && !errors.length && !messages.length) {
    box.hidden = true;
    box.innerHTML = '';
    setConfigStatus('valid');
    return;
  }
  box.hidden = false;
  box.className = errors.length ? 'notice error' : (warnings.length ? 'notice warning' : 'notice info');
  const errorHtml = errors.map(e => `<li>${escapeHtml(e)}</li>`).join('');
  const warningHtml = warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('');
  const messageHtml = messages.map(m => `<li>${escapeHtml(m)}</li>`).join('');
  box.innerHTML = `${errors.length ? `<strong>Errors</strong><ul>${errorHtml}</ul>` : ''}${warnings.length ? `<strong>Warnings</strong><ul>${warningHtml}</ul>` : ''}${messages.length ? `<strong>Messages</strong><ul>${messageHtml}</ul>` : ''}`;
  setConfigStatus(errors.length ? 'invalid' : (warnings.length ? 'warning' : 'valid'));
}

function readElementValue(el) {
  if (el.dataset.customInput && el.value === '__custom__') {
    return ($(el.dataset.customInput)?.value || '').trim();
  }
  return el.value;
}

function fieldLabel(el) {
  return requiredFields.get(el.name) || el.closest('label')?.childNodes[0]?.textContent?.trim() || el.name;
}

function clientValidationErrors() {
  const errors = [];
  for (const el of form.elements) {
    if (!el.name || el.disabled) continue;
    const label = fieldLabel(el);
    if (requiredFields.has(el.name) && readElementValue(el) === '') {
      errors.push(`${label} is required.`);
      continue;
    }
    if (el.type === 'number' && el.value !== '' && !el.validity.valid) {
      if (el.validity.rangeUnderflow) errors.push(`${label} must be at least ${el.min}.`);
      else if (el.validity.rangeOverflow) errors.push(`${label} must be at most ${el.max}.`);
      else if (el.validity.stepMismatch) errors.push(`${label} must use a valid increment.`);
      else errors.push(`${label} must be a valid number.`);
    }
  }
  return errors;
}

function getFormProfile() {
  const data = {...current};
  for (const el of form.elements) {
    if (!el.name) continue;
    if (checkboxFields.has(el.name)) data[el.name] = el.checked;
    else if (intFields.has(el.name)) data[el.name] = el.value === '' ? null : parseInt(el.value, 10);
    else if (floatFields.has(el.name)) data[el.name] = el.value === '' ? null : parseFloat(el.value);
    else data[el.name] = readElementValue(el);
  }
  data.id = current?.id || crypto.randomUUID();
  data.ui_mode = $('ui-mode').value;
  data.speculative_enabled = Boolean(data.speculative_method);
  return data;
}

function syncCustomSelect(el) {
  if (!el.dataset.customInput) return;
  const custom = $(el.dataset.customInput);
  if (!custom) return;
  custom.hidden = el.value !== '__custom__';
  custom.disabled = el.value !== '__custom__';
}

function syncGenerationConfig() {
  const path = $('generation-config-path');
  if (!path) return;
  path.hidden = $('generation-config').value !== 'custom';
  path.disabled = $('generation-config').value !== 'custom';
}

function applyMode() {
  const mode = $('ui-mode').value || 'wizard';
  document.body.dataset.mode = mode;
  updateSummary();
  syncExpertOptions(mode);
  if (mode === 'expert') {
    const activeTab = document.querySelector('.tab.active')?.dataset.tab || 'basic';
    setActivePage(activeTab);
  } else {
    setActivePage(wizardSteps[currentWizardIndex] || 'basic');
  }
  updateWizardNav();
}

function syncExpertOptions(mode = $('ui-mode').value || 'wizard') {
  const expertMode = mode === 'expert';
  document.querySelectorAll('option[data-expert-option]').forEach(option => {
    option.hidden = !expertMode;
    option.disabled = !expertMode;
  });
  const deployment = form.elements.deployment_mode;
  if (!expertMode && deployment?.selectedOptions?.[0]?.dataset.expertOption !== undefined) {
    deployment.value = 'local';
  }
}

function setActivePage(tab) {
  const wizardIndex = wizardSteps.indexOf(tab);
  if (wizardIndex >= 0) currentWizardIndex = wizardIndex;
  document.querySelectorAll('.tab-page').forEach(page => page.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tab));
  document.querySelectorAll('.wizard-step').forEach((btn, index) => {
    const isActive = btn.dataset.wizardTab === tab;
    btn.classList.toggle('active', isActive);
    btn.classList.toggle('complete', index < currentWizardIndex);
  });
  const target = $('tab-' + tab);
  if (target) target.classList.add('active');
  updateWizardNav();
  if (tab === 'review') updateWizardReview();
}

function updateWizardNav() {
  const nav = $('wizard-nav');
  if (!nav) return;
  const prev = $('wizard-prev');
  const next = $('wizard-next');
  prev.disabled = currentWizardIndex === 0;
  next.hidden = currentWizardIndex === wizardSteps.length - 1;
  next.textContent = currentWizardIndex === wizardSteps.length - 2 ? 'Review' : 'Next';
}

function updateWizardReview() {
  const profile = getFormProfile();
  const summary = [
    ['Profile', profile.name],
    ['Model', profile.model],
    ['Endpoint', `${profile.host}:${profile.port}`],
    ['dtype', profile.dtype || 'auto'],
    ['Quantization', profile.quantization || 'none'],
    ['Parallelism', `TP ${profile.tensor_parallel_size || 1}, PP ${profile.pipeline_parallel_size || 1}`],
    ['Backend', profile.distributed_executor_backend || 'auto'],
    ['Speculative', profile.speculative_method || 'disabled'],
  ];
  $('wizard-summary').innerHTML = summary.map(([label, value]) => `
    <div class="review-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>`).join('');
  $('wizard-command-box').textContent = $('command-box').textContent || '';
  $('wizard-script-box').textContent = 'Click Download startup script to export the shell script.';
}

function setConfigStatus(state) {
  const el = $('config-status');
  if (!el) return;
  const labels = {valid: 'Valid', warning: 'Review warnings', invalid: 'Invalid', checking: 'Checking'};
  el.textContent = labels[state] || labels.checking;
  el.className = `config-status ${state || 'checking'}`;
}

function updateSummary(profile = getFormProfile()) {
  if (!$('summary-mode')) return;
  $('summary-mode').textContent = ($('ui-mode').value || 'wizard') === 'expert' ? 'Expert' : 'Wizard';
  $('summary-endpoint').textContent = `${profile.host || 'HOST'}:${profile.port || 'PORT'}`;
  $('summary-parallel').textContent = `TP ${profile.tensor_parallel_size || 1} / PP ${profile.pipeline_parallel_size || 1}`;
  $('summary-speculative').textContent = profile.speculative_method || 'Off';
}

function syncSpeculativeFields() {
  const method = readElementValue($('speculative-method'));
  document.querySelectorAll('.spec-field').forEach(el => { el.hidden = !method; });
  document.querySelectorAll('.spec-ngram').forEach(el => { el.hidden = !['ngram', 'ngram_gpu'].includes(method); });
  const modelUseful = ['draft_model', 'eagle', 'eagle3', 'custom_class', 'mtp', 'mlp_speculator', 'dflash'].includes(method) || method.endsWith('_mtp');
  document.querySelectorAll('.spec-model').forEach(el => { el.hidden = !modelUseful; });
  document.querySelectorAll('.spec-draft-tp').forEach(el => { el.hidden = !['draft_model', 'eagle', 'eagle3'].includes(method); });
  const help = {
    '': 'Speculative decoding is disabled.',
    draft_model: 'Draft-model speculative decoding needs a smaller compatible auxiliary model.',
    eagle: 'EAGLE often uses a compatible speculator model/head. Leave the model blank only for model-native or extra JSON configurations your vLLM version supports.',
    eagle3: 'EAGLE3 often uses a compatible speculator model/head. Leave the model blank only for model-native or extra JSON configurations your vLLM version supports.',
    ngram: 'N-gram is easy to enable and does not need a separate draft model.',
    ngram_gpu: 'GPU n-gram does not need a separate draft model.',
    suffix: 'Suffix decoding does not need a separate draft model.',
    mtp: 'Use MTP when the target model or checkpoint supports MTP-style speculative decoding.',
    dflash: 'Use DFlash only with compatible target models and vLLM versions.',
    custom_class: 'Custom proposer classes are experimental and require a class path in the model field.'
  };
  $('speculative-help').textContent = help[method] || 'Confirm this speculative method is supported by your installed vLLM version.';
}

function setFormProfile(p) {
  current = JSON.parse(JSON.stringify(p));
  $('ui-mode').value = p.ui_mode === 'beginner' ? 'wizard' : (p.ui_mode || 'wizard');
  for (const el of form.elements) {
    if (!el.name || !(el.name in p)) continue;
    if (checkboxFields.has(el.name)) {
      el.checked = Boolean(p[el.name]);
    } else if (el.dataset.customInput) {
      const value = p[el.name] ?? '';
      const hasOption = Array.from(el.options).some(option => option.value === value);
      el.value = hasOption ? value : '__custom__';
      const custom = $(el.dataset.customInput);
      if (custom) custom.value = hasOption ? '' : value;
      syncCustomSelect(el);
    } else {
      el.value = p[el.name] ?? '';
    }
  }
  syncGenerationConfig();
  syncSpeculativeFields();
  syncExpertOptions();
  applyMode();
  markDirty(false);
}

async function loadProfiles(selectId = null) {
  const body = await api('/api/profiles');
  profiles = body.profiles;
  $('profile-select').innerHTML = profiles.map(p => `<option value="${p.id}">${escapeHtml(p.name || p.model)}</option>`).join('');
  const id = selectId || profiles[0]?.id;
  if (id) $('profile-select').value = id;
  setFormProfile(profileById($('profile-select').value));
  await updateCommandPreview();
}

function escapeHtml(s) { return String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

async function saveProfile() {
  const errors = clientValidationErrors();
  if (errors.length) {
    showValidation({errors});
    throw new Error(errors.join('\n'));
  }
  const profile = getFormProfile();
  validateJsonFields(profile);
  const body = await api('/api/profiles', {method: 'POST', body: JSON.stringify({profile})});
  showValidation(body);
  await loadProfiles(body.profile.id);
  markDirty(false);
  return body.profile;
}

async function updateCommandPreview() {
  const errors = clientValidationErrors();
  updateSummary();
  if (errors.length) {
    $('command-box').textContent = 'Fix validation errors to preview the command.';
    $('command-meta').textContent = 'Preview paused';
    setConfigStatus('invalid');
    showValidation({errors});
    return;
  }
  const profile = getFormProfile();
  try {
    const body = await api('/api/profiles/command', {method: 'POST', body: JSON.stringify({profile})});
    $('command-box').textContent = body.shell || 'Fix validation errors to preview the command.';
    $('command-meta').textContent = body.ok ? `${(body.cmd || []).length} argv item(s), generated safely without shell execution` : 'Preview unavailable';
    $('speculative-json-box').textContent = body.speculative_json ? JSON.stringify(body.speculative_json, null, 2) : '';
    showValidation(body);
    await updateRunbook(profile);
    if ($('tab-review')?.classList.contains('active')) updateWizardReview();
  } catch (e) {
    $('command-box').textContent = 'Fix validation errors to preview the command.';
    $('command-meta').textContent = 'Preview unavailable';
    setConfigStatus('invalid');
    $('speculative-json-box').textContent = '';
    setMessage(e.message);
  }
}

function validateJsonFields(profile) {
  for (const field of jsonFields) {
    try {
      const parsed = JSON.parse(profile[field] || '{}');
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
        throw new Error('must be a JSON object');
      }
    } catch (e) {
      throw new Error(`${field} must be valid JSON object syntax: ${e.message}`);
    }
  }
}

async function updateRunbook(profile = getFormProfile()) {
  try {
    const body = await api('/api/cluster/runbook', {method: 'POST', body: JSON.stringify({profile})});
    const text = Object.entries(body.sections || {})
      .map(([title, lines]) => `${title}\n${(lines || []).join('\n')}`)
      .join('\n\n');
    $('runbook-box').textContent = text;
  } catch {
    if ($('runbook-box')) $('runbook-box').textContent = 'Fix validation errors to generate the cluster runbook.';
  }
}

function schedulePreview() {
  markDirty(true);
  updateSummary();
  setConfigStatus('checking');
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updateCommandPreview, 250);
}

async function previewCommand() {
  await updateCommandPreview();
}

async function startServer() {
  const errors = clientValidationErrors();
  if (errors.length) {
    showValidation({errors});
    throw new Error(errors.join('\n'));
  }
  const profile = getFormProfile();
  validateJsonFields(profile);
  const body = await api('/api/start', {method:'POST', body: JSON.stringify({profile})});
  $('command-box').textContent = body.command;
  await refreshStatus();
}

async function restartServer() {
  const p = await saveProfile();
  const body = await api(`/api/profiles/${p.id}/restart`, {method:'POST'});
  $('command-box').textContent = body.command;
  await refreshStatus();
}

async function stopServer() {
  await api('/api/stop', {method:'POST'});
  await refreshStatus();
}

async function refreshStatus() {
  try {
    const s = await api('/api/status');
    const pill = $('server-status');
    pill.className = 'status-pill ' + (s.running ? 'running' : 'stopped');
    pill.textContent = s.running ? `vLLM server: running PID ${s.process?.pid || ''} - ${s.uptime_seconds}s` : (s.exit_code !== null && s.exit_code !== undefined ? `vLLM server: stopped - exit ${s.exit_code}` : 'vLLM server: stopped');
    if (s.dashboard?.warning) showValidation({dashboard: s.dashboard});
  } catch (e) {
    $('server-status').className = 'status-pill error';
    $('server-status').textContent = e.message;
  }
}

async function refreshLogs() {
  try {
    const text = await (await fetch('/api/logs')).text();
    const box = $('log-box');
    if (box.textContent !== text) {
      box.textContent = text;
      box.scrollTop = box.scrollHeight;
    }
  } catch {}
}

async function refreshGpu() {
  const body = await api('/api/gpu');
  renderGpu(body);
}

async function sendTest() {
  const body = await api('/api/chat', {method:'POST', body: JSON.stringify({
    prompt: $('test-prompt').value,
    max_tokens: parseInt($('test-max-tokens').value || '128', 10),
    temperature: parseFloat($('test-temperature').value || '0.2')
  })});
  $('response-box').textContent = JSON.stringify(body, null, 2);
}

async function exportText(kind) {
  const profile = getFormProfile();
  if (kind === 'spark' && profile.deployment_mode !== 'spark_ray') {
    throw new Error('Spark/Ray script export is only needed when deployment mode is Spark-managed Ray.');
  }
  const p = await saveProfile();
  const text = await (await fetch(`/api/export/${kind}/${p.id}`)).text();
  $('command-box').textContent = text;
}

async function exportStartupScript() {
  const errors = clientValidationErrors();
  if (errors.length) {
    showValidation({errors});
    throw new Error(errors.join('\n'));
  }
  const profile = getFormProfile();
  validateJsonFields(profile);
  const text = await api('/api/start-script', {method: 'POST', body: JSON.stringify({profile})});
  $('command-box').textContent = text;
  const wizardBox = $('wizard-script-box');
  if (wizardBox) {
    wizardBox.textContent = text;
    wizardBox.focus();
    const range = document.createRange();
    range.selectNodeContents(wizardBox);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
  showValidation({messages: ['Startup script generated.']});
  return text;
}

async function downloadStartupScript() {
  const text = await exportStartupScript();
  downloadText('start-vllm.sh', text, 'text/x-shellscript');
  showValidation({messages: ['Startup script downloaded as start-vllm.sh.']});
}

async function runAction(action) {
  try {
    await action();
  } catch (e) {
    setMessage(e.message);
  }
}

async function copyCommand(sourceId = 'command-box') {
  const text = $(sourceId).textContent || '';
  if (!text.trim()) return;
  try {
    if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable');
    await navigator.clipboard.writeText(text);
  } catch {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
  }
  setMessage('Command copied to clipboard.', 'warning');
}

function applyPreset(name) {
  const set = (field, value) => {
    const el = form.elements[field];
    if (!el) return;
    if (checkboxFields.has(field)) el.checked = Boolean(value);
    else el.value = value;
  };
  if (name === 'single_gpu') {
    set('deployment_mode', 'local'); set('tensor_parallel_size', 1); set('pipeline_parallel_size', 1); set('distributed_executor_backend', '');
  } else if (name === 'single_node_multi_gpu') {
    set('deployment_mode', 'single_node_multi_gpu'); set('tensor_parallel_size', 4); set('pipeline_parallel_size', 1); set('distributed_executor_backend', 'mp');
  } else if (name === 'ray_multi_node') {
    set('deployment_mode', 'ray_cluster'); set('distributed_executor_backend', 'ray'); set('ray_num_nodes', 2); set('ray_gpus_per_node', 8); set('tensor_parallel_size', 8); set('pipeline_parallel_size', 2);
  } else if (name === 'moe_expert_parallel') {
    set('deployment_mode', 'expert_parallel_moe'); set('is_moe_model', true); set('enable_expert_parallel', true); set('enable_ep_weight_filter', true); set('distributed_executor_backend', 'ray');
  } else if (name === 'low_latency_spec') {
    set('speculative_method', 'ngram'); set('num_speculative_tokens', 4); set('prompt_lookup_min', 2); set('prompt_lookup_max', 5); set('performance_mode', 'interactivity');
  } else if (name === 'high_throughput') {
    set('performance_mode', 'throughput'); set('max_num_seqs', 256); set('enable_chunked_prefill', true);
  } else if (name === 'memory_saving') {
    set('kv_cache_dtype', 'fp8'); set('gpu_memory_utilization', 0.85); set('cpu_offload_gb', 0);
  }
  syncSpeculativeFields();
  schedulePreview();
}

function downloadJson(filename, payload) {
  downloadText(filename, JSON.stringify(payload, null, 2), 'application/json');
}

function downloadText(filename, text, type = 'text/plain') {
  const blob = new Blob([text], {type});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function exportProfiles(scope) {
  if (scope === 'current') await saveProfile();
  const url = scope === 'all' ? '/api/profiles/export' : `/api/profiles/${activeProfileId()}/export`;
  const payload = await api(url);
  const name = scope === 'all' ? 'vllm-profiles.json' : `vllm-profile-${activeProfileId()}.json`;
  downloadJson(name, payload);
}

async function importProfiles(file) {
  if (!file) return;
  const payload = JSON.parse(await file.text());
  const body = await api('/api/profiles/import', {method: 'POST', body: JSON.stringify(payload)});
  await loadProfiles(body.imported[0]?.id);
  showValidation({warnings: [`Imported ${body.count} profile(s).`]});
}

function renderGpu(body) {
  const box = $('gpu-box');
  if (!body.ok) {
    box.innerHTML = `<div class="empty-state">${escapeHtml(body.error || 'GPU telemetry unavailable')}</div>`;
    return;
  }
  const summary = body.summary || {};
  const rows = (body.gpus || []).map(g => `
    <tr>
      <td>${escapeHtml(g.index ?? '')}</td>
      <td>${escapeHtml(g.name || '')}</td>
      <td>${escapeHtml(g.gpu_utilization_percent ?? '')}%</td>
      <td>${escapeHtml(g.memory_used_mb ?? '')} / ${escapeHtml(g.memory_total_mb ?? '')} MB (${escapeHtml(g.memory_percent ?? '')}%)</td>
      <td>${escapeHtml(g.temperature_c ?? '')} C</td>
      <td>${escapeHtml(g.power_draw_w ?? '')} / ${escapeHtml(g.power_limit_w ?? '')} W</td>
      <td>${escapeHtml(g.pstate || '')}</td>
    </tr>`).join('');
  const procs = (body.processes || []).map(p => `
    <tr>
      <td>${escapeHtml(p.pid ?? '')}</td>
      <td>${escapeHtml(p.process_name || '')}</td>
      <td>${escapeHtml(p.used_memory_mb ?? '')} MB</td>
      <td>${escapeHtml(p.gpu_uuid || '')}</td>
    </tr>`).join('');
  box.innerHTML = `
    <div class="telemetry-summary">
      <span>${escapeHtml(summary.gpu_count ?? 0)} GPU(s)</span>
      <span>${escapeHtml(summary.memory_used_mb ?? 0)} / ${escapeHtml(summary.memory_total_mb ?? 0)} MB</span>
      <span>Max util ${escapeHtml(summary.max_gpu_utilization_percent ?? '')}%</span>
      <span>Max temp ${escapeHtml(summary.max_temperature_c ?? '')} C</span>
    </div>
    <table class="data-table">
      <thead><tr><th>ID</th><th>GPU</th><th>Util</th><th>Memory</th><th>Temp</th><th>Power</th><th>P-state</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="7">No GPUs reported.</td></tr>'}</tbody>
    </table>
    <table class="data-table">
      <thead><tr><th>PID</th><th>Process</th><th>GPU memory</th><th>GPU UUID</th></tr></thead>
      <tbody>${procs || '<tr><td colspan="4">No compute processes reported.</td></tr>'}</tbody>
    </table>`;
}

function bind() {
  form.addEventListener('input', schedulePreview);
  form.addEventListener('change', (e) => {
    if (e.target?.dataset?.customInput) syncCustomSelect(e.target);
    if (e.target?.id === 'generation-config') syncGenerationConfig();
    if (e.target?.id === 'speculative-method') syncSpeculativeFields();
    if (e.target?.name === 'deployment_mode') syncExpertOptions();
    schedulePreview();
  });
  $('ui-mode').addEventListener('change', () => { applyMode(); schedulePreview(); });
  $('preset-select').addEventListener('change', (e) => { applyPreset(e.target.value); e.target.value = ''; });
  document.querySelectorAll('[data-custom-input]').forEach(syncCustomSelect);
  $('profile-select').addEventListener('change', () => { setFormProfile(profileById(activeProfileId())); updateCommandPreview(); });
  $('save-profile').addEventListener('click', () => runAction(async () => { await saveProfile(); await updateCommandPreview(); }));
  $('start-server').addEventListener('click', () => runAction(startServer));
  $('restart-server').addEventListener('click', () => runAction(restartServer));
  $('stop-server').addEventListener('click', () => runAction(stopServer));
  $('new-profile').addEventListener('click', () => runAction(async () => {
    const p = {...current, id: crypto.randomUUID(), name: 'New profile', model: current?.model || 'Qwen/Qwen3-8B'};
    current = p; setFormProfile(p); await saveProfile();
  }));
  $('clone-profile').addEventListener('click', () => runAction(async () => {
    const body = await api(`/api/profiles/${activeProfileId()}/clone`, {method:'POST'});
    await loadProfiles(body.profile.id);
  }));
  $('delete-profile').addEventListener('click', async () => {
    if (!confirm('Delete this profile?')) return;
    await runAction(async () => { await api(`/api/profiles/${activeProfileId()}`, {method:'DELETE'}); await loadProfiles(); });
  });
  $('export-systemd').addEventListener('click', () => runAction(() => exportText('systemd')));
  $('export-ray').addEventListener('click', () => runAction(() => exportText('ray')));
  $('export-spark').addEventListener('click', () => runAction(() => exportText('spark')));
  $('export-startup-script').addEventListener('click', () => runAction(exportStartupScript));
  $('wizard-download-script').addEventListener('click', () => runAction(downloadStartupScript));
  $('wizard-start-server').addEventListener('click', () => runAction(startServer));
  $('wizard-prev').addEventListener('click', () => {
    if (currentWizardIndex > 0) setActivePage(wizardSteps[currentWizardIndex - 1]);
  });
  $('wizard-next').addEventListener('click', () => runAction(async () => {
    const errors = clientValidationErrors();
    if (errors.length) {
      showValidation({errors});
      throw new Error(errors.join('\n'));
    }
    await updateCommandPreview();
    if (currentWizardIndex < wizardSteps.length - 1) setActivePage(wizardSteps[currentWizardIndex + 1]);
  }));
  document.querySelectorAll('.wizard-step').forEach(btn => btn.addEventListener('click', () => runAction(async () => {
    const targetIndex = wizardSteps.indexOf(btn.dataset.wizardTab);
    if (targetIndex > currentWizardIndex) {
      const errors = clientValidationErrors();
      if (errors.length) {
        showValidation({errors});
        throw new Error(errors.join('\n'));
      }
      await updateCommandPreview();
    }
    setActivePage(btn.dataset.wizardTab);
  })));
  $('export-profile').addEventListener('click', () => runAction(() => exportProfiles('current')));
  $('export-all-profiles').addEventListener('click', () => runAction(() => exportProfiles('all')));
  $('copy-command').addEventListener('click', () => runAction(() => copyCommand('command-box')));
  $('copy-wizard-command').addEventListener('click', () => runAction(() => copyCommand('wizard-command-box')));
  $('import-profile-file').addEventListener('change', async (e) => {
    await runAction(async () => { await importProfiles(e.target.files[0]); e.target.value = ''; });
  });
  $('import-profiles').addEventListener('click', () => $('import-profile-file').click());
  $('refresh-gpu').addEventListener('click', () => runAction(refreshGpu));
  $('send-test').addEventListener('click', () => runAction(sendTest));
  document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => setActivePage(btn.dataset.tab)));
}

setupTooltips();
bind();
loadVllmOptions().then(() => loadProfiles()).then(refreshStatus).then(refreshGpu);
setInterval(refreshStatus, 3000);
setInterval(refreshLogs, 2000);
