list = dir('E:\1_task\2021kfb\??_*');   
for j =1:size(list,1)
    a = list(j).name;
    b = dir(['E:\\1_task\\2021kfb\\',a,'\\Block_ACTH_0.625\\?.png']);
    name = a;
    for i = 1:size(b,1) 
        mkdir('E:\\2_task\\DIRNet-PyTorch-master\\lhy_dataset\\CRCS');
        mkdir('E:\\2_task\\DIRNet-PyTorch-master\\lhy_dataset\\CRCS\\fixed_images');
        mkdir('E:\\2_task\\DIRNet-PyTorch-master\\lhy_dataset\\CRCS\\moving_images');
        mkdir('E:\\2_task\\DIRNet-PyTorch-master\\lhy_dataset\\CRCS\\fixed_images\\img_align_celeba');
        mkdir('E:\\2_task\\DIRNet-PyTorch-master\\lhy_dataset\\CRCS\\moving_images\\img_align_celeba');
        copyfile(sprintf(['E:\\1_task\\2021kfb\\',name,'\\Block_ACTH_5\\%d.png'],i),sprintf(['E:\\2_task\\DIRNet-PyTorch-master\\lhy_dataset\\CRCS\\fixed_images\\img_align_celeba\\',name,'_cluster%d.png'],i));
        copyfile(sprintf(['E:\\1_task\\2021kfb\\',name,'\\Block_Tpit_5\\%d.png'],i),sprintf(['E:\\2_task\\DIRNet-PyTorch-master\\lhy_dataset\\CRCS\\moving_images\\img_align_celeba\\',name,'_cluster%d.png'],i));
        

    end
end