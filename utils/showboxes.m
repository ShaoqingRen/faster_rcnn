function showboxes(im, boxes, legends, color_conf)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes)
%
% -------------------------------------------------------

fix_width = 800;
if isa(im, 'gpuArray')
    im = gather(im);
end
imsz = size(im);
scale = fix_width / imsz(2);
im = imresize(im, scale);

if size(boxes{1}, 2) >= 5
    boxes = cellfun(@(x) [x(:, 1:4) * scale, x(:, 5)], boxes, 'UniformOutput', false);
else
    boxes = cellfun(@(x) x(:, 1:4) * scale, boxes, 'UniformOutput', false);
end

if ~exist('color_conf', 'var')
    color_conf = 'default';
end

image(im); 
axis image;
axis off;
set(gcf, 'Color', 'white');

valid_boxes = cellfun(@(x) ~isempty(x), boxes, 'UniformOutput', true);
valid_boxes_num = sum(valid_boxes);

if valid_boxes_num > 0
    switch color_conf
        case 'default'
            colors_candidate = colormap('hsv');
            colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/valid_boxes_num)):end, :);
            colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
            colors = cell(size(valid_boxes));
            colors(valid_boxes) = colors_candidate(1:sum(valid_boxes));
        case 'voc'
            colors_candidate = colormap('hsv');
            colors_candidate = colors_candidate(1:(floor(size(colors_candidate, 1)/20)):end, :);
            colors_candidate = mat2cell(colors_candidate, ones(size(colors_candidate, 1), 1))';
            colors = colors_candidate;
    end
            

    for i = 1:length(boxes)
        if isempty(boxes{i})
            continue;
        end

        for j = 1:size(boxes{i})
            box = boxes{i}(j, 1:4);
            if size(boxes{i}, 2) >= 5
                score = boxes{i}(j, end);
                linewidth = 2 + min(max(score, 0), 1) * 2;
                rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors{i});
                label = sprintf('%s : %.3f', legends{i}, score);
                text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
            else
                linewidth = 2;
                rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors{i});
                label = sprintf('%s(%d)', legends{i}, i);
                text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
            end
        end

    end
end
end

function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)

rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
end

