function [ACCURACY] = SRBFR(numTrainSamples, filePath)
f=dir(filePath);
c=1;
l=length(f)-3;
A=zeros(l,32256);
flag=zeros(l+3,100);
for i=4:l+3
    filelist=dir(fullfile(filePath,f(i).name));
    a=length(filelist)-5;
    for j=1:numTrainSamples
        s=floor(5+a*rand);
        flag(i,s)=1;
        u=imread(filelist(s).name);
        A(c,:)=reshape(u,1,[]);           
        c=c+1;
    end    
end
average=mean(A);
standard_deviation=std(A);
for i=1:length(A(1,:))
    A(:,i)=(A(:,i)-average(i))/standard_deviation(i);
end

[COEFF,SCORE,latent]=pca(A);
c=0;
s=sum(latent);
for r=1:l*numTrainSamples
    c=c+latent(r);
    if c/s>=0.95 
        break;
    end
end

B=SCORE(:,1:r)';
c=0;a=0;
s=zeros(l,1);
for i=4:l+3
    filelist=dir(fullfile(filePath,f(i).name));
    for j=5:length(filelist)-1
        if flag(i,j)==0
            a=a+1;
            u=imread(filelist(j).name);
            test=double(reshape(u,1,[]));
            for k=1:length(test)
                test(k)=(test(k)-average(k))/standard_deviation(k);
            end
            y=test*COEFF(:,1:r);
            y=y';
            
            lambda=0.00005;
            init_x=zeros(l*numTrainSamples,1);
            
            [x]=feature_sign(B,y,lambda,init_x);
            for k=1:l
                s(k)=sum(x(numTrainSamples*(k-1)+1:numTrainSamples*k));
            end
            [~,I]=max(s);
            if i-3==I
                c=c+1;
            end
        end
    end
end
ACCURACY=c/a;
return;
