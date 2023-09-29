
list = dir('E:\2_task\dataset_CRCS\??_*');   
for j =1:size(list,1)
    a = list(j).name;
    name = a;
    b = dir(['E:\2_task\dataset_CRCS\',name,'\fixed*.png']);
    for i = 1:size(b,1)
        image_name = b(i).name;
        im = imread(['E:\2_task\dataset_CRCS\',name,'\\',image_name]);
        if ndims(im) == 2
                gray=im;
        else
            gray=rgb2gray(im);
        end

       
        std=stdfilt(gray,strel('disk',4).Neighborhood);

        
        sep=imbinarize(rescale(std));
      
        sep(1:15,:)=false;
        sep(end-15:end,:)=false;
        sep(:,1:15)=false;
        sep(:,end-15:end)=false;


        sep = imclose(sep,strel('disk',7));
       
        sep=imfill(sep,'holes'); 
       
        cc=bwconncomp(sep);

        numPixels = cellfun(@numel,cc.PixelIdxList);
        idx = find(numPixels <= 400); 
        for i_idx=idx(1:end)
            sep(cc.PixelIdxList{i_idx}) = 0;
        end

        
        equal = mean(gray(~sep), 'all');
        if ~isfile(['E:\2_task\dataset_CRCS\',name,'\\equal_fixed.txt'])
            writematrix(equal, ['E:\2_task\dataset_CRCS\',name,'\\equal_fixed.txt']);
        end
    end
end