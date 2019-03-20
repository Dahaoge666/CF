import numpy as np
import random
import math
from operator import itemgetter
class AvgEvaluate():
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0


        self.totalSet = {}
        self.total_len = 0      #总评分数
        self.total_rating = 0.0   #总评分和
        # 用户相似度矩阵
        self.matrix_inverted = {} #倒排矩阵
        self.movie_privacy = dict()
        self.movie_count = 0   #电影数量
        self.rating_max = 5.0
        self.rating_min = 0.5
        self.userNum = 610
        self.matrix_privacy = {}
        self.epcilon = 1

    def laplace_get(self):
        laplace_num = int((5.0-0.5)/self.epcilon)#此处epcilon               
        loc, scale = 0., 1.
        s = np.random.laplace(loc, scale, self.epcilon)
        ss=s[0]
        return ss

    def totalCount(self,filename):
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            self.total_len += 1
            self.totalSet.setdefault(user, {})
            self.totalSet[user][movie] = rating
            self.total_rating += float(rating)
        print(self.total_len)

        # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)
    


    def matrixInverted(self):
        print('Building movie-user table ...')
        movie_rating = {}
        for user, movies in self.totalSet.items():
            for movie in movies:
                if movie not in movie_rating:
                    movie_rating[movie] = set()
                movie_rating[movie].add(self.totalSet[user][movie])
        print('Build movie-user table success!')
        self.matrix_inverted = movie_rating
        self.movie_count = len(movie_rating)
        print('Total movie number = %d' % self.movie_count)


    def avgWithoutPrivacy(self):
        # avg=self.total_rating/self.total_len
        # print(avg)
        i = 0
        for movie,ratings in self.matrix_inverted.items():
            i += 1#控制循环次数
            if i<11:
                rating_total = 0.0
                for rating in ratings:
                    rating_total += float(rating)
                rating_avg = rating_total/len(ratings)
                print(rating_avg)
            else:
                break
        print("-----------------------------")

    def avgWithPrivacy(self):
        i = 0
        laplaca_value = self.laplace_get()
        for movie,ratings in self.matrix_inverted.items():
            i += 1#控制循环次数
            if i<11:
                
                rating_total = 0.0
                rating_list = list(ratings)
                maxrating = minrating = float(rating_list[0])
                for rating in ratings:
                    if float(rating)>maxrating:
                        maxrating = float(rating)
                    if float(rating)<minrating:
                        minrating = float(rating)
                    rating_total += float(rating)
                rating_avg = (rating_total+laplaca_value)/len(ratings)
                print(rating_avg)
            else:
                break

    def globalAvg(self):
        laplace_value = self.laplace_get()
        GlobalAvg = (self.total_rating + laplace_value)/self.total_len
        return GlobalAvg
 
    def itemAvg(self):
        betai = 10
        laplace_value=self.laplace_get()
        for movie,ratings in self.matrix_inverted.items():
            j = 0#读数量
            rating_total = 0.0
            for rating in ratings:
                rating_total += float(rating)
                j += 1
            ItemAvg = (rating_total+betai*self.globalAvg()+laplace_value)/(j+betai)    
            self.movie_privacy[movie] = ItemAvg              
        print('Build movie-user table success!')

    







 # 读文件得到“用户-电影”数据
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            
            if random.random() < pivot:
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = str(float(rating)-self.movie_privacy[movie])
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = str(float(rating)-self.movie_privacy[movie])
                testSet_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)




    # 计算用户之间的相似度
    def calc_user_sim(self):
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.movie_count = len(movie_user)
        print('Total movie number = %d' % self.movie_count)

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movies matrix success!')

        # 计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')


    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainSet[v]:
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]


    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        N = self.n_rec_movie
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user, in enumerate(self.trainSet):
            test_movies = self.testSet.get(user, {})
            rec_movies = self.recommend(user)
            # print(rec_movies)    打印相似度
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))









if __name__ == "__main__":
    rating_file = 'E:/code/recommend/data/ml-latest-small/ratings.csv'
    avg=AvgEvaluate()
    avg.totalCount(rating_file)
    avg.matrixInverted()
    avg.globalAvg()
    avg.itemAvg()

    avg.get_dataset(rating_file)
    avg.calc_user_sim()
    avg.evaluate()
    # avg.userAvg()
