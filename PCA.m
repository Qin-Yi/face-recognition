function [COEFF,SCORE,latent] = PCA(A)
%[U,S,V]=svd(A); maybe this way is better...
    mu=mean(A);
    [m,n]=size(A);
    covariance=zeros(n);
    for i=1:n
        for j=1:i
            covariance(i,j)=(A(:,i)-mu(i))'/m*(A(:,j)-mu(j));
            covariance(j,i)=covariance(i,j);
        end
    end
    [V,D]=eig(covariance);
    [latent,index]=sort(diag(D),'descend');
    COEFF=V(:,index);
    SCORE=A/COEFF';
return;
