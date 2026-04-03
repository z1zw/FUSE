function fig11_standalone_publish()
close all
clc

outDir = fullfile(pwd, 'figures_standalone_fig11_matlab');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

accent = [40 120 181] / 255;
teal = [27 153 139] / 255;
green = [42 157 143] / 255;
orange = [231 111 81] / 255;
gold = [244 162 97] / 255;
purple = [122 92 250] / 255;
mint = [128 237 153] / 255;
slate = [52 73 94] / 255;
cyan = [72 191 227] / 255;
gray = [125 133 151] / 255;
bg = [245 247 251] / 255;
panelA = [238 243 251] / 255;
panelB = [255 246 236] / 255;
panelC = [247 240 255] / 255;

schedLabels = categorical({'Kappa-key','Template','Valid Fragments'});
schedValues = [99.19156922188537, 99.46585823588855, 31.961888263317455];

platforms = categorical({'Multi-core','FUSE'});
latencyValues = [1.037164, 0.32844];
idleValues = [12.714723487897684, 12.880028963832242];

summaryMetrics = categorical({'Scheduling latency','Idle ratio','Buffer ops'});
summaryMulti = [100.0, 100.0, 100.0];
summaryFuse = [31.66712303936504, 101.30011066376632, 100.0];
summaryIdeal = [6.859731601431083, 0.0, 4.994927027237962];

zkpToCnnLatency = 0.02944;
cnnToAesLatency = 0.03335;
avgRecovery = 4.704592333333333;

f1 = figure('Color', 'w', 'Position', [80 80 1640 620]);
t = tiledlayout(f1, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(t, 1);
ax1.Color = panelA;
b1 = bar(ax1, schedLabels, schedValues, 0.62, 'FaceColor', 'flat', 'EdgeColor', [0.13 0.19 0.25], 'LineWidth', 1.0);
b1.CData = [accent; teal; green];
ylim(ax1, [0 110]);
ylabel(ax1, 'Match ratio (%)', 'FontWeight', 'bold');
xlabel(ax1, 'Fragment filter', 'FontWeight', 'bold');
title(ax1, '(a) TARS fragment scheduling', 'FontWeight', 'bold');
grid(ax1, 'on');
ax1.GridAlpha = 0.18;
ax1.GridLineStyle = '--';
for i = 1:numel(schedValues)
    text(ax1, i, schedValues(i) + 2.0, sprintf('%.1f%%', schedValues(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end

ax2 = nexttile(t, 2);
ax2.Color = panelB;
yyaxis(ax2, 'left');
x2 = 1:numel(platforms);
b2 = bar(ax2, x2 - 0.18, latencyValues, 0.34, 'FaceColor', slate, 'EdgeColor', [0.13 0.19 0.25], 'LineWidth', 1.0);
ylabel(ax2, 'Scheduling latency (ms)', 'FontWeight', 'bold');
ylim(ax2, [0 max(latencyValues) * 1.35]);
yyaxis(ax2, 'right');
b3 = bar(ax2, x2 + 0.18, idleValues, 0.34, 'FaceColor', orange, 'EdgeColor', [0.13 0.19 0.25], 'LineWidth', 1.0);
ylabel(ax2, 'Idle ratio (%)', 'FontWeight', 'bold');
ylim(ax2, [0 max(idleValues) * 1.22]);
ax2.XTick = x2;
ax2.XTickLabel = cellstr(platforms);
xlabel(ax2, 'Runtime platform', 'FontWeight', 'bold');
title(ax2, '(b) Switch latency and idle ratio', 'FontWeight', 'bold');
grid(ax2, 'on');
ax2.GridAlpha = 0.18;
ax2.GridLineStyle = '--';
yyaxis(ax2, 'left');
for i = 1:numel(latencyValues)
    text(ax2, x2(i) - 0.18, latencyValues(i) + max(0.02, latencyValues(i) * 0.05), sprintf('%.2f', latencyValues(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end
yyaxis(ax2, 'right');
for i = 1:numel(idleValues)
    text(ax2, x2(i) + 0.18, idleValues(i) + max(0.2, idleValues(i) * 0.04), sprintf('%.1f%%', idleValues(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end

ax3 = nexttile(t, 3);
ax3.Color = panelC;
x3 = 1:numel(summaryMetrics);
hold(ax3, 'on');
bar(ax3, x3 - 0.24, summaryMulti, 0.22, 'FaceColor', slate, 'EdgeColor', [0.13 0.19 0.25], 'LineWidth', 1.0);
bar(ax3, x3, summaryFuse, 0.22, 'FaceColor', orange, 'EdgeColor', [0.13 0.19 0.25], 'LineWidth', 1.0);
bar(ax3, x3 + 0.24, summaryIdeal, 0.22, 'FaceColor', mint, 'EdgeColor', [0.13 0.19 0.25], 'LineWidth', 1.0);
hold(ax3, 'off');
ax3.XTick = x3;
ax3.XTickLabel = cellstr(summaryMetrics);
ax3.XTickLabelRotation = 8;
ylim(ax3, [0 120]);
ylabel(ax3, 'Normalized cost (% of multi-core)', 'FontWeight', 'bold');
xlabel(ax3, 'Overhead metric', 'FontWeight', 'bold');
title(ax3, '(c) Multi-core vs FUSE vs ideal', 'FontWeight', 'bold');
grid(ax3, 'on');
ax3.GridAlpha = 0.18;
ax3.GridLineStyle = '--';
for i = 1:numel(summaryMetrics)
    text(ax3, x3(i) - 0.24, summaryMulti(i) + 2.4, sprintf('%.1f%%', summaryMulti(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    text(ax3, x3(i), summaryFuse(i) + 2.4, sprintf('%.1f%%', summaryFuse(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    text(ax3, x3(i) + 0.24, summaryIdeal(i) + 2.4, sprintf('%.1f%%', summaryIdeal(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
end

lg1 = legend(t, {'Kappa-key hits','Template hits','Valid fragments','Multi-core','FUSE','Ideal standalone'}, 'Orientation', 'horizontal', 'NumColumns', 3, 'Location', 'southoutside');
lg1.Box = 'off';
title(t, 'Fig. 11. Cross-Task Scheduling Overhead Reduction', 'FontWeight', 'bold', 'FontSize', 16);
exportgraphics(f1, fullfile(outDir, 'fig11_switch_overheads_matlab.png'), 'Resolution', 320);

f2 = figure('Color', 'w', 'Position', [120 120 1560 640]);
t2 = tiledlayout(f2, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax4 = nexttile(t2, 1);
style_graph_axis(ax4, bg);
title(ax4, '(d) ZKP -> CNN switch', 'FontWeight', 'bold');
draw_node(ax4, 0.18, 0.62, 0.08, 'ZKP', gray, [1 1 1]);
draw_node(ax4, 0.78, 0.62, 0.08, 'CNN', green, [1 1 1]);
draw_node(ax4, 0.48, 0.28, 0.09, 'State', purple, [1 1 1]);
draw_edge(ax4, [0.26 0.55], [0.42 0.34], '--', '');
draw_edge(ax4, [0.54 0.34], [0.72 0.55], '-', '');
text(ax4, 0.5, 0.06, sprintf('FUSE switch latency: %.2f ms', zkpToCnnLatency), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'BackgroundColor', [1.0 0.965 0.847], 'Margin', 4);

ax5 = nexttile(t2, 2);
style_graph_axis(ax5, bg);
title(ax5, '(e) CNN -> AES switch', 'FontWeight', 'bold');
draw_node(ax5, 0.18, 0.62, 0.08, 'CNN', gray, [1 1 1]);
draw_node(ax5, 0.78, 0.62, 0.08, 'AES', orange, [1 1 1]);
draw_node(ax5, 0.48, 0.28, 0.09, 'State', purple, [1 1 1]);
draw_edge(ax5, [0.26 0.55], [0.42 0.34], '--', '');
draw_edge(ax5, [0.54 0.34], [0.72 0.55], '-', '');
text(ax5, 0.5, 0.06, sprintf('FUSE switch latency: %.2f ms', cnnToAesLatency), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'BackgroundColor', [1.0 0.965 0.847], 'Margin', 4);

ax6 = nexttile(t2, 3);
style_graph_axis(ax6, bg);
title(ax6, '(f) Full state inheritance', 'FontWeight', 'bold');
draw_node(ax6, 0.20, 0.72, 0.075, 'ZKP', accent, [1 1 1]);
draw_node(ax6, 0.50, 0.82, 0.075, 'CNN', green, [1 1 1]);
draw_node(ax6, 0.80, 0.72, 0.075, 'AES', orange, [1 1 1]);
draw_node(ax6, 0.50, 0.46, 0.09, 'State', purple, [1 1 1]);
draw_node(ax6, 0.32, 0.18, 0.065, 'Buf', gold, [0 0 0]);
draw_node(ax6, 0.68, 0.18, 0.065, 'Cache', cyan, [0 0 0]);
draw_edge(ax6, [0.27 0.67], [0.44 0.52], '-', sprintf('%.1f%%', summaryFuse(1)));
draw_edge(ax6, [0.50 0.74], [0.50 0.59], '-', '');
draw_edge(ax6, [0.73 0.67], [0.56 0.52], '-', sprintf('%.1f ms', avgRecovery));
draw_edge(ax6, [0.44 0.36], [0.36 0.25], '-', '');
draw_edge(ax6, [0.56 0.36], [0.64 0.25], '-', '');
text(ax6, 0.5, 0.92, sprintf('Ideal latency cost: %.1f%%', summaryIdeal(1)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'Color', [0.24 0.25 0.36]);
text(ax6, 0.5, 0.06, sprintf('FUSE buffer cost: %.1f%% | ideal: %.1f%%', summaryFuse(3), summaryIdeal(3)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10, 'BackgroundColor', [0.851 0.973 0.871], 'Margin', 4);

lg2 = legend(t2, {'ZKP task','CNN task','AES task','Shared state','Buffer','Cache','Completed path'}, 'Orientation', 'horizontal', 'NumColumns', 4, 'Location', 'southoutside');
lg2.Box = 'off';
title(t2, 'Fig. 11d-f. Cross-Task State Reuse Patterns', 'FontWeight', 'bold', 'FontSize', 16);
exportgraphics(f2, fullfile(outDir, 'fig11_state_paths_matlab.png'), 'Resolution', 320);
end

function style_graph_axis(ax, bg)
axis(ax, [0 1 0 1])
axis(ax, 'off')
ax.Color = bg;
end

function draw_node(ax, x, y, r, label, color, textColor)
rectangle(ax, 'Position', [x - r, y - r, 2 * r, 2 * r], 'Curvature', [1 1], 'FaceColor', color, 'EdgeColor', [0.13 0.14 0.16], 'LineWidth', 1.6)
text(ax, x, y, label, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'FontSize', 10, 'Color', textColor)
end

function draw_edge(ax, p0, p1, lineStyle, labelText)
annotationLine(ax, p0(1), p0(2), p1(1), p1(2), lineStyle);
if ~isempty(labelText)
    text(ax, (p0(1) + p1(1)) / 2, (p0(2) + p1(2)) / 2 + 0.05, labelText, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'FontSize', 9)
end
end

function annotationLine(ax, x0, y0, x1, y1, lineStyle)
plot(ax, [x0 x1], [y0 y1], 'Color', [0.13 0.14 0.16], 'LineWidth', 1.8, 'LineStyle', lineStyle)
dx = x1 - x0;
dy = y1 - y0;
quiver(ax, x1 - 0.001 * dx, y1 - 0.001 * dy, 0.001 * dx, 0.001 * dy, 0, 'Color', [0.13 0.14 0.16], 'LineWidth', 1.8, 'MaxHeadSize', 10)
end
