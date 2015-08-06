function vis_label(imdb, roidb)

    rois = roidb.rois;
    for iIM = 1:length(rois)
        im = imread(imdb.image_at(iIM));
        boxes = arrayfun(@(x) rois(iIM).boxes(rois(iIM).class == x, :), 1:length(imdb.classes), 'UniformOutput', false);
        legends = imdb.classes;
        showboxes(im, boxes, legends);
        pause;
    end
end
  