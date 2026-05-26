function [dxker, dyker] = ker_lcrt(image, mx, my)
    % Kernal of matrix B of Riesz Transform --.--Yang
    b1 = mx(1,2);
    b2 = my(1,2);
    [m, n] = size(image);
    
    % 初始化 dxker 和 dyker
    dxker = zeros(m, n);
    dyker = zeros(m, n);
    
for xi=1:m
     for xj=1:n
         dxker(xi,xj)=-1*1i*(xi/b1)/(sqrt((xi/b1)^2+(xj/b2)^2));
     end
end
 
for yi=1:m
     for yj=1:n
         dyker(yi,yj)=-1*1i*(yj/b2)/(sqrt((yi/b1)^2+(yj/b2)^2));
     end
 end


% function [dxker,dyker]= ker_lcrt(image,mx,my)
% % Kernal of matrix B of Riesz Transform --.--Yang
% b1=mx(1,2);
% b2=my(1,2);
% [m,n] = size(image);
% 
% 
%     
% for xi=-m/2:m/2-1
%      for xj=-n/2:n/2-1
%          dxker(xi,xj)=-1*1i*(xi/b1)/(sqrt((xi/b1)^2+(xj/b2)^2));
%      end
% end
%  
% for yi=-m/2:m/2-1
%      for yj=-n/2:n/2-1
%          dyker(yi,yj)=-1*1i*(yj/b2)/(sqrt((yi/b1)^2+(yj/b2)^2));
%      end
%  end
% for xi=1:m
%      for xj=1:n
%          dxker(xi,xj)=-1*1i*(xi/b1)/(sqrt((xi/b1)^2+(xj/b2)^2));
%      end
% end
%  
% for yi=1:m
%      for yj=1:n
%          dyker(yi,yj)=-1*1i*(yj/b2)/(sqrt((yi/b1)^2+(yj/b2)^2));
%      end
%  end