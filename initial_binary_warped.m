
list = dir('E:\1_task\2021kfb\??_*');   
for j =1:size(list,1)
    a = list(j).name;
    b = dir(['E:\\1_task\\2021kfb\\',a,'\\Block_ACTH_0.625\\?.png']);
    name = a;
    for i = 1:size(b,1) 
        tform_file = sprintf(['E:\\2_task\\dataset_CRCS\\',name,'_cluster%d\\tform.mat'],i);
        load(tform_file);
        
        fixed_image = imread(sprintf(['E:\\2_task\\dataset_CRCS\\',name,'_cluster%d\\fixed.png'],i));
        moving_image = imread(sprintf(['E:\\2_task\\dataset_CRCS\\',name,'_cluster%d\\moving.png'],i));
        
        imref = imref2d(size(fixed_image));
            
        binary_fixed = binary_segmentation(fixed_image);
        binary_moving = binary_segmentation(moving_image);
        
        binary_initial_moving = imwarp(binary_moving,tform,'OutputView',imref);
        
        imwrite(binary_initial_moving,sprintf(['E:\\2_task\\dataset_CRCS\\',name,'_cluster%d\\binary_initial_moving.png'],i));
        imwrite(binary_fixed,sprintf(['E:\\2_task\\dataset_CRCS\\',name,'_cluster%d\\binary_fixed.png'],i));
        imwrite(binary_moving,sprintf(['E:\\2_task\\dataset_CRCS\\',name,'_cluster%d\\binary_moving.png'],i));
    end
end

function binary_image = binary_segmentation(im)
        if ndims(im) == 2
                gray=im;
        else
            gray=rgb2gray(im);
        end
        
        % 方差：
        std=stdfilt(gray,strel('disk',4).Neighborhood);

        
        sep=imbinarize(rescale(std));
        
                            
        sep(1:15,:)=false;
        sep(end-15:end,:)=false;
        sep(:,1:15)=false;
        sep(:,end-15:end)=false;

        
        sep = imerode(sep,strel('disk',2));
        sep = imclose(sep,strel('disk',5));
        
        sep=imfill(sep,'holes'); 
       
        
        
        cc=bwconncomp(sep);
        
        numPixels = cellfun(@numel,cc.PixelIdxList);
        idx = find(numPixels <= 400); 
        for i_idx=idx(1:end)
            sep(cc.PixelIdxList{i_idx}) = 0;
        end

        
        gray(~sep) = 0;
        gray(sep) = 255;
        binary_image = gray;
end