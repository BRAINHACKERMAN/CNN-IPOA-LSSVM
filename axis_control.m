function axis_control(flag)
% 用以控制坐标参数
image_size = 400; % 图像大小
if flag==1
    ylabel('Frequency (Hz)')
    xlabel('x')
else
    % 无边，无框，无坐标轴
    box off; axis off % 无框
    set(gcf,'Position',[600 100 image_size image_size]) % 图像大小
    set(gca,'xticklabel',[],'yticklabel',[]); % 无坐标轴
    set(gca,'position',[0 0 1 1]) % 无边
end

end