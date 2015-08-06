function [ rectsLTRB ] = RectLTWH2LTRB(rectsLTWH)
%rects (l, t, r, b) to (l, t, w, h)

rectsLTRB = [rectsLTWH(:, 1), rectsLTWH(:, 2), rectsLTWH(:, 1)+rectsLTWH(:,3)-1, rectsLTWH(:,2)+rectsLTWH(:,4)-1];
end

