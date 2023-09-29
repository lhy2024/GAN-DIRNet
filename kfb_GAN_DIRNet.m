
list = dir('E:\\2_task\\dataset_CRCS\\??_*');
 
app.first_line = 1;
app.order = 2;
MrTRE = zeros(1,size(list,1));
MxrTRE = zeros(1,size(list,1));
ArTRE = zeros(1,size(list,1));
num = 0;  

for j =1:size(list,1)
    a = list(j).name;
    name = a;
    
    num = num + 1; 
    Tpit_path = ['E:\\2_task\\dataset_CRCS\\',a,'\\moving.png'];
    Block_Tpit = imread(Tpit_path);
    diagonal = sqrt(size(Block_Tpit,1)^2 + size(Block_Tpit,2)^2);  
        
    
    ACTH_path = ['E:\\2_task\\dataset_CRCS\\',a,'\\fixed.png'];
    Block_ACTH = imread(ACTH_path);
    
    V_path = ['E:\\2_task\\dataset_CRCS\\',a,'\\GAN_DIRNet\\V_%d.mat'];  
    V = load(V_path);
        
    landmark_file = ['E:\\2_task\\dataset_CRCS\\',a,'\\landmark.txt'];
    LM = readmatrix(landmark_file);
    rTRE = zeros(1,size(LM,1)); 
        
    for s = 1:size(LM,1) 
       ACTH_red_x = LM(s,1); 
       ACTH_red_y = LM(s,2); 
       Tpit_green_x = LM(s,3);
       Tpit_green_y = LM(s,4);
           
       
       if size(Block_Tpit,1)>= size(Block_Tpit,2)
           Tpit_green_x = min(180,max(1,round(Tpit_green_x*180/size(Block_Tpit,1))));
           Tpit_green_y = min(180,max(1,round(Tpit_green_y*180/size(Block_Tpit,1))));  
       else
           Tpit_green_x = min(256,max(1,round(Tpit_green_x*256/size(Block_Tpit,2))));
           Tpit_green_y = min(256,max(1,round(Tpit_green_y*256/size(Block_Tpit,2))));
       end
      
       if size(Block_ACTH,1)>= size(Block_ACTH,2)
           ACTH_red_x = min(180,max(1,round(ACTH_red_x*180/size(Block_ACTH,1))));
           ACTH_red_y = min(180,max(1,round(ACTH_red_y*180/size(Block_ACTH,1))));
       else
           ACTH_red_x = min(256,max(1,round(ACTH_red_x*256/size(Block_ACTH,2))));
           ACTH_red_y = min(256,max(1,round(ACTH_red_y*256/size(Block_ACTH,2))));
       end
           
      
       Tpit_output_y = Tpit_green_y + V.V(2,Tpit_green_x,Tpit_green_y); 
       Tpit_output_x = Tpit_green_x + V.V(1,Tpit_green_x,Tpit_green_y); 
       
           
       rTRE(s) = sqrt((Tpit_output_x-ACTH_red_x)^2+(Tpit_output_y-ACTH_red_y)^2)/diagonal; 
    end
    [sorted_rTRE,~] = sort(rTRE(1:size(LM,1))); 
    if mod(size(LM,1),2)
        MrTRE(num) = sorted_rTRE((size(LM,1)+1)/2);
    else
        MrTRE(num) = sorted_rTRE(size(LM,1)/2);
    end
    MxrTRE(num) = max(rTRE(:));
    ArTRE(num) = mean(rTRE(:));
end
    


AMrTRE = mean(MrTRE(:));
AMxrTRE = mean(MxrTRE(:));
AArTRE = mean(ArTRE(:));

[sorted_MrTRE,~] = sort(MrTRE(1:num)); 
if mod(num,2)
    MMrTRE = sorted_MrTRE((num+1)/2);
else
    MMrTRE = sorted_MrTRE(num/2);
end

[sorted_MxrTRE,~] = sort(MxrTRE(1:num)); 
if mod(num,2)
    MMxrTRE = sorted_MxrTRE((num+1)/2);
else
    MMxrTRE = sorted_MxrTRE(num/2);
end

[sorted_ArTRE,~] = sort(ArTRE(1:num)); 
if mod(num,2)
    MArTRE = sorted_ArTRE((num+1)/2);
else
    MArTRE = sorted_ArTRE(num/2);
end


ColumnName = ["AMrTRE","AMxrTRE","AArTRE","MMrTRE","MMxrTRE","MArTRE"]';
table = array2table([AMrTRE,AMxrTRE,AArTRE,MMrTRE,MMxrTRE,MArTRE],'RowNames',{'DIRNet'},'VariableNames',ColumnName);
if app.first_line    
    writetable(table,'E:\\2_task\\criteria\\DIRNet_all.xlsx','WriteRowNames',true);
    app.first_line = 0;    
else
    app.order = app.order + 1;   
    if app.order < 10
        writetable(table,'E:\\2_task\\criteria\\DIRNet_all.xlsx','WriteRowNames',true,'WriteVariableNames',false,'sheet','Sheet1','Range',sprintf('A%d',app.order));
    elseif app.order < 100
        writetable(table,'E:\\2_task\\criteria\\DIRNet_all.xlsx','WriteRowNames',true,'WriteVariableNames',false,'sheet','Sheet1','Range',sprintf('A%2d',app.order));
    elseif app.order < 1000
        writetable(table,'E:\\2_task\\criteria\\DIRNet_all.xlsx','WriteRowNames',true,'WriteVariableNames',false,'sheet','Sheet1','Range',sprintf('A%3d',app.order));
    end
end
