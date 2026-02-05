%% new_compile_whole_python_v2.m
% Purpose:
%   Build the `whole` struct and generate the Word report using the
%   Python-derived metrics + figures you downloaded locally.
%
% What you need to edit:
%   1) Set `base_dir` below to your local folder that contains:
%        - sub-XXXX_ses-Y_all_metrics.mat
%        - task folders (rest / sts / valsalva / breathing) with figures
%
% Notes:
%   - Valsalva section: ONLY reports the Valsalva ratio (per your update).
%   - STS: expects metrics stored in sts/sts_metrics.mat (variables like mean_HR_sup, mean_HR_plt, dHR, mean_MAP_sup, mean_MAP_plt, dMAP, ...)
%   - Table styling is matched to your original script (FormalTable header style + borders).

clear; clc;

%% =========================
%  USER CONFIG (EDIT THIS)
%  =========================
base_dir = 'C:\Users\15355\Desktop\sub-2010\ses-1';  % <-- EDIT ME

%% =========================
%  LOAD COMBINED METRICS
%  =========================
% Expected: <base_dir>\sub-XXXX_ses-Y_all_metrics.mat
bundle_path = fullfile(base_dir, sprintf('%s_%s_all_metrics.mat', get_sub_id(base_dir), get_ses_id(base_dir)));
if ~exist(bundle_path, 'file')
    % Fallback: find any *_all_metrics.mat under base_dir
    cands = dir(fullfile(base_dir, '*_all_metrics.mat'));
    if isempty(cands)
        error('Could not find *_all_metrics.mat under base_dir: %s', base_dir);
    end
    bundle_path = fullfile(base_dir, cands(1).name);
end

bundle = load(bundle_path);
if ~isfield(bundle, 'metrics_by_task')
    error('The combined bundle does not contain `metrics_by_task`. File: %s', bundle_path);
end

tasks = bundle.metrics_by_task;
whole = struct();

%% =========================
%  REST (ECG + BP)
%  =========================
if isfield(tasks, 'rest')
    rest = tasks.rest;

    % ECG/HRV
    whole.mean_RR      = firstField(rest, {'mean_RR','mean_rr'}, NaN);                         % s
    whole.mean_HR      = firstField(rest, {'mean_HR','mean_hr'}, NaN);                         % bpm
    whole.RMSSD        = firstField(rest, {'RMSSD_ms','RMSSD','rmssd_ms'}, NaN);               % ms
    whole.LF_HF_ratio  = firstField(rest, {'LF_HF','LF_HF_ratio','lf_hf','LF_HF_power'}, NaN); % ratio

    % BP
    whole.mean_MAP     = firstField(rest, {'mean_MAP','mean_map'}, NaN);
    whole.mean_sysBP   = firstField(rest, {'mean_sysBP','mean_sysbp','mean_sbp'}, NaN);
    whole.mean_diaBP   = firstField(rest, {'mean_diaBP','mean_diabp','mean_dbp'}, NaN);
    whole.mean_pulseBP = firstField(rest, {'mean_pulseBP','mean_pulsebp'}, NaN);
else
    warning('No rest metrics found in bundle.');
end

%% =========================
%  STS (supine vs plateau)
%  =========================
% Your Python STS metrics file contains fields like:
%   mean_HR_sup, mean_HR_plt, dHR, mean_MAP_sup, mean_MAP_plt, dMAP, ...
% and is saved as sts/sts_metrics.mat
if isfield(tasks, 'sts')
    sts = tasks.sts;

    % HR
    whole.baseline_HR = firstField(sts, {'baseline_HR','baseline_hr','mean_HR_sup','mean_hr_sup','mean_HR_supine'}, NaN);
    whole.plateau_HR  = firstField(sts, {'plateau_HR','plateau_hr','mean_HR_plt','mean_hr_plt','mean_HR_plateau'}, NaN);
    whole.delta_HR    = firstField(sts, {'delta_HR','delta_hr','dHR','dhr'}, NaN);

    % MAP
    whole.baseline_BP = firstField(sts, {'baseline_BP','baseline_bp','baseline_MAP','baseline_map','mean_MAP_sup','mean_map_sup'}, NaN);
    whole.plateau_BP  = firstField(sts, {'plateau_BP','plateau_bp','plateau_MAP','plateau_map','mean_MAP_plt','mean_map_plt'}, NaN);
    whole.delta_BP    = firstField(sts, {'delta_BP','delta_bp','delta_MAP','delta_map','dMAP','dmap'}, NaN);

else
    warning('No STS metrics found in bundle.');
end

%% =========================
%  VALSALVA (ratio only)
%  =========================
if isfield(tasks, 'valsalva')
    val = tasks.valsalva;
    whole.Valsalva_ratio = firstField(val, {'valsalva_ratio','Valsalva_ratio','valsalvaRatio'}, NaN);
else
    warning('No valsalva metrics found in bundle.');
end

%% =========================
%  BREATHING (E/I ratio + delta)
%  =========================
if isfield(tasks, 'breathing')
    br = tasks.breathing;

    whole.delta_HR_responses = firstField(br, {'diff_bpm','delta_bpm','HR_diff','hr_diff','HR_delta_RSA','HR_delta'}, NaN);
    whole.E_I_ratio          = firstField(br, {'ratio','E_I_ratio','ei_ratio','EIRatio'}, NaN);
else
    warning('No breathing metrics found in bundle.');
end

%% =========================
%  SAVE compile_whole.mat
%  =========================
out_mat = fullfile(base_dir, 'compile_whole.mat');
save(out_mat, 'whole');
fprintf('[OK] Saved %s\n', out_mat);

%% =========================
%  FIGURE PATHS
%  =========================
% Valsalva figure (your pipeline outputs something like valsalva_best_rep_hr.png)
val_fig_path = pickFirstExisting({
    fullfile(base_dir, 'valsalva', 'valsalva_best_rep_hr.png')
    fullfile(base_dir, 'valsalva', 'valsalva_best_rep_hr.jpg')
    fullfile(base_dir, 'valsalva', 'valsalva_best_rep_hr.tif')
});

% STS figure name update: you said it is now STS_HR_MAP (not *_plot)
sts_fig_path = pickFirstExisting({
    fullfile(base_dir, 'sts', 'STS_HR_MAP.png')
    fullfile(base_dir, 'sts', 'STS_HR_MAP.PNG')
    fullfile(base_dir, 'sts', 'STS_HR_MAP_plot.png')
    fullfile(base_dir, 'sts', 'STS_HR_MAP_plot.PNG')
    fullfile(base_dir, 'sts', 'STS_HR_MAP_plot.jpg')
});

% Breathing figure (your pipeline outputs one of these)
breath_fig_path = pickFirstExisting({
    fullfile(base_dir, 'breathing', 'deep_breathing_HR_plot.png')
    fullfile(base_dir, 'breathing', 'deep_breathing_HR_plot.PNG')
    fullfile(base_dir, 'breathing', 'breathing_hr_8to9min.png')
    fullfile(base_dir, 'breathing', 'breathing_hr_8to9min.PNG')
    fullfile(base_dir, 'breathing', 'breathing_hr_window.png')
    fullfile(base_dir, 'breathing', 'breathing_hr_plot.png')
});

%% =========================
%  GENERATE REPORT
%  =========================
import mlreportgen.dom.*
import myReports.SessionReport

rpt = SessionReport('Subject1_Report', 'docx');

% --- placeholders (customize as needed) ---
rpt.Age = '45';
rpt.Height = '170 cm';
rpt.Weight = '75 kg';
rpt.Date = datestr(now, 'mmm dd, yyyy HH:MM');
rpt.randomtest = 'Passed';
rpt.ReactionTime = '325 ms';
rpt.congruency = 'High';
rpt.InhaleVolume = 'Yes';

%% -------- Rest HRV table (style matched) --------
header = {'RR Property', 'Value', 'Unit'};
rr_data = {
    'Mean RR',         sprintf('%.4f', whole.mean_RR),      's';
    'Mean HR',         sprintf('%.2f', whole.mean_HR),      'bpm';
    'RMSSD*',          sprintf('%.2f', whole.RMSSD),        'ms';
    'LF/HF Ratio*',    sprintf('%.2f', whole.LF_HF_ratio),  '-';
};

AT = FormalTable(header, rr_data);
AT = applyOriginalTableStyle(AT);
rpt.Mytable = AT;

%% -------- Rest BP table (style matched) --------
header2 = {'BP Property', 'Value', 'Unit'};
BP_data = {
    'Mean ABP',        sprintf('%.2f', whole.mean_MAP),     'mmHg';
    'systolic ABP',    sprintf('%.2f', whole.mean_sysBP),   'mmHg';
    'Diastolic ABP',   sprintf('%.2f', whole.mean_diaBP),   'mmHg';
    'Pulse Pressure',  sprintf('%.2f', whole.mean_pulseBP), 'mmHg';
};

AT2 = FormalTable(header2, BP_data);
AT2 = applyOriginalTableStyle(AT2);
rpt.Table2 = AT2;

%% -------- Valsalva figure + ratio --------
if ~isempty(val_fig_path)
    img = Image(val_fig_path);
    img.Width = '3.5in';
    img.Height = '2in';
    rpt.valfigure = img;
else
    warning('Valsalva figure not found under base_dir.');
end

valText = Text('Valsalva Ratio');
valText.Bold = true;
valText.FontFamilyName = 'Arial';
valText.FontSize = '12pt';
rpt.Valratio = valText;

rpt.valvalue = sprintf('%.2f', whole.Valsalva_ratio);

% Valsalva table: ONLY ratio (per your update)
header5 = {'Val Property', 'Value', 'Unit'};
val_data = {
    'Valsalva Ratio*', sprintf('%.2f', whole.Valsalva_ratio), '';
};
AT5 = FormalTable(header5, val_data);
AT5 = applyOriginalTableStyle(AT5);
rpt.Table5 = AT5;

%% -------- STS figure + table --------
if ~isempty(sts_fig_path)
    img = Image(sts_fig_path);
    img.Width = '3.7 in';
    img.Height = '2.4in';
    rpt.STSplots = img;
else
    warning('STS figure not found under base_dir.');
end

header3 = {'Measurements', 'Value', 'Unit'};
STS_data = {
    'baseline HR',   sprintf('%.2f', whole.baseline_HR), 'BPM';
    'Plateau HR',    sprintf('%.2f', whole.plateau_HR),  'BPM';
    'Baseline BP',   sprintf('%.2f', whole.baseline_BP), 'mmHg';
    'Plateau BP',    sprintf('%.2f', whole.plateau_BP),  'mmHg';
    'Delta HR*',     sprintf('%.2f', whole.delta_HR),    'BPM';
    'Delta BP*',     sprintf('%.2f', whole.delta_BP),    'mmHg';
};
AT3 = FormalTable(header3, STS_data);
AT3 = applyOriginalTableStyle(AT3);
rpt.STSTable = AT3;

%% -------- Breathing: ratio + delta + figure --------
% Your template expects strings here
rpt.EIratio  = sprintf('%.2f', whole.E_I_ratio);
rpt.DeltaHR  = sprintf('%.2f', whole.delta_HR_responses);

if ~isempty(breath_fig_path)
    img = Image(breath_fig_path);
    img.Width = '3.5in';
    img.Height = '2.3in';
    rpt.DPBfig = img;
else
    warning('Breathing figure not found under base_dir.');
end


%% ---------- spirometry --------------------

%% ---------- spirometry (from derived .mat) --------------------
import mlreportgen.dom.*

spiro_mat = fullfile(base_dir, 'spirometry', 'spirometry_metrics.mat');

if exist(spiro_mat, 'file')
    S = load(spiro_mat);

    % These field names match the extractor script I suggested
    fev1 = getfield_or_nan(S, 'FEV1_max');
    fvc  = getfield_or_nan(S, 'FVC_max');
    pef  = getfield_or_nan(S, 'PEF_max');

    % You said you want FVC/FEV1 computed from extracted values
    fvc_over_fev1 = nan;
    if isfinite(fvc) && isfinite(fev1) && fev1 ~= 0
        fvc_over_fev1 = fvc / fev1;
    end

    header_spiro = {'Spirometry index', 'Value', 'Unit'};
    sprio_data = {
        'FEV1',      sprintf('%.2f', fev1),         'liters';
        'FVC',       sprintf('%.2f', fvc),          'liters';
        'FVC/FEV1',  sprintf('%.2f', fvc_over_fev1),'NA';
        'PEF',       sprintf('%.2f', pef),          'L/s';
    };

else
    % If spirometry missing, show blanks instead of erroring
    header_spiro = {'Spirometry index', 'Value', 'Unit'};
    sprio_data = {
        'FEV1',      'NA', 'liters';
        'FVC',       'NA', 'liters';
        'FVC/FEV1',  'NA', 'NA';
        'PEF',       'NA', 'L/s';
    };
end

AT_spiro = FormalTable(header_spiro, sprio_data);
AT_spiro = applyOriginalTableStyle(AT_spiro);   % reuse your helper style
rpt.sprio_Table = AT_spiro;



%% Finalize
fill(rpt);
rptview('Subject1_Report.docx');

%% =========================
%  Helper functions
%  =========================
function v = firstField(s, names, defaultVal)
    v = defaultVal;
    if isempty(s) || ~isstruct(s); return; end
    for i = 1:numel(names)
        nm = names{i};
        if isfield(s, nm)
            vv = s.(nm);
            v = toScalar(vv);
            return;
        end
    end
end

function x = toScalar(v)
    % Convert MATLAB loaded values (possibly arrays/cells) to a scalar double where possible.
    if iscell(v) && ~isempty(v)
        v = v{1};
    end
    if ischar(v) || isstring(v)
        x = str2double(v);
        if isnan(x)
            x = v;
        end
        return;
    end
    if isnumeric(v)
        if isempty(v)
            x = NaN;
        else
            x = double(v(1));
        end
        return;
    end
    x = v;
end

function p = pickFirstExisting(paths)
    p = '';
    for i = 1:numel(paths)
        if exist(paths{i}, 'file')
            p = paths{i};
            return;
        end
    end
end

function sub_id = get_sub_id(base_dir)
    toks = regexp(base_dir, '(sub-\d+)', 'tokens', 'once');
    if ~isempty(toks)
        sub_id = toks{1};
    else
        sub_id = 'sub-XXXX';
    end
end

function ses_id = get_ses_id(base_dir)
    toks = regexp(base_dir, '(ses-\d+)', 'tokens', 'once');
    if ~isempty(toks)
        ses_id = toks{1};
    else
        ses_id = 'ses-X';
    end
end

function T = applyOriginalTableStyle(T)
    import mlreportgen.dom.*   % <<< ADD THIS LINE

    headerRowStyle = {
        InnerMargin("2pt","2pt","2pt","2pt"), ...
        Bold(false), ...
        RowSep('solid'), ...
        Border('solid')
    };

    HeaderRow = T.Header;
    HeaderRow.Style = headerRowStyle;

    T.Style = { Border('single') };

    % These mimic your original code (safe to keep)
    try
        T.Style{1,1}.LeftStyle  = 'none';
        T.Style{1,1}.RightStyle = 'none';
    catch
    end

    try
        T.Header.RowSepWidth = '1pt';
    catch
    end
end
function v = getfield_or_nan(S, name)
    if isstruct(S) && isfield(S, name)
        v = S.(name);
        if ischar(v) || isstring(v), v = str2double(v); end
        if isempty(v), v = nan; end
        if numel(v) > 1, v = v(1); end
    else
        v = nan;
    end
end

