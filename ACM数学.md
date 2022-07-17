# 一、数论

## 1.基础数论

### 试除法分解质因数

$O(log(n))$

```c++
	auto divide = [&](int x){
        vector<pair<int, int>> res;
        for(int i = 2; i <= x / i; i ++){
            if(x % i == 0){
                int s = 0;
                while(x % i == 0) x /= i,  s ++;
                res.push_back({i, s});
            }
        }
        if(x > 1) res.push_back({x, 1});
        for(auto [ke, va] : res)
            cout << ke << ' ' << va << "\n";
        cout << "\n";
    };
```

***



### 线性筛法

$O(\sqrt n)$

```c++
	vector<int> primes;
    vector<bool> st(n + 1);
    auto isPrime = [&](int x){
        for(int i = 2; i <= x; i ++){
            if(!st[i]) primes.push_back(i);
            for(int j = 0; primes[j] <= x / i; j ++){
                st[primes[j] * i] = true;
                if(i % primes[j] == 0) break;
            }
        }
    };
```

***



### 约数

- **试除法求所有约数$O(\sqrt n)$**

```c++
auto getDivisors = [&](int x){
        vector<int> res;
        for(int i = 1; i <= x / i; i ++){
            if(x % i == 0){
                res.push_back(i);
                if(x / i != i) res.push_back(x / i);
            }
        }
        
        sort(res.begin(), res.end());
        return res;
    };
```

- **约数个数**

  ```c++
  /*
  AcWing 870.约数个数
  给定 n 个正整数 ai，请你输出这些数的乘积的约数个数，答案对 10^9+7 取模。
  数据范围
  1≤n≤100,
  1≤ai≤2×10^9
  */
  #include <bits/stdc++.h>
  
  using namespace std;
  
  const int mod = 1e9 + 7;
  
  int main()
  {
      int n;
      cin >> n;
       
      unordered_map<int, int> res; 
      while(n --){
          int x;
          cin >> x;
          
          for(int i = 2; i <= x / i; i ++){
              while(x % i == 0){
                  x /= i;
                  res[i] ++;
              }
          }
          if(x > 1) res[x] ++;
      }
      
      long long t = 1;
      for(auto item : res) t = t * (item.second + 1) % mod;
      
      cout << t << endl;
      
      return 0;
  }
  ```

- **约数之和**

  ```c++
  /*
  AcWing 871. 约数之和
  给定 n 个正整数 ai，请你输出这些数的乘积的约数之和，答案对 10^9+7 取模。
  数据范围
  1≤n≤100,
  1≤ai≤2×10^9
  */
  
  #include <bits/stdc++.h>
  
  using namespace std;
  
  const int mod = 1e9 + 7;
  
  int main()
  {
      int n;
      cin >> n;
       
      unordered_map<int, int> res; 
      while(n --){
          int x;
          cin >> x;
          
          for(int i = 2; i <= x / i; i ++){
              while(x % i == 0){
                  x /= i;
                  res[i] ++;
              }
          }
          if(x > 1) res[x] ++;
      }
      
      long long t = 1;
      for(auto item : res){
          long long s = 1;
          int p = item.first, a = item.second;
          for(int i = 0; i < a; i ++) s = (s * p + 1) % mod;
          
          t = t * s % mod;    
      }
      
      cout << t << endl;
      
      return 0;
  }
  ```

- 最大公约数

  ```c++
  function<int(int, int)> gcd = [&](int a, int b){
          return b ? gcd(b, a % b) : a;
      };
  ```

***




### 欧拉函数

- **欧拉函数**
  $$
  N = P_1 ^ {a_1} * P_2 ^ {a_2} * ...* P_k ^ {a_k}
  $$
  
  $$
  \phi(N) = N * (1 - \frac{1}{P_1}) * (1 - \frac{1}{P_2}) * ...* (1 - \frac{1}{P_k})
  $$
  $O(\sqrt n)$

  ```c++
  int Euler(int x)
  {   
      int res = x;
       for(int i = 2; i <= x / i; i ++)
          if(x % i == 0){
              res = res / i * (i - 1);
              while(x % i == 0) x /= i;
          }
      if(x > 1) res = res / x * (x - 1); 
      
      return res;
  }
  ```

  

- 筛法求欧拉函数

  $O(n)$

  ```c++
  //求1~n中所有数的欧拉函数之和
  const int N = 1000010;
  int primes[N], cnt;
  int phi[N];
  bool st[N];
  
  LL getEuler(int n)
  {
      phi[1] = 1;
      for(int i = 2; i <= n; i ++){
          if(!st[i]){
              primes[cnt ++] = i;
              phi[i] = i - 1;
          }
          for(int j = 0; primes[j] <= n / i; j ++){
              st[i * primes[j]] = true;
              if(i % primes[j] == 0){
                  phi[i * primes[j]] = phi[i] * primes[j];
                  break;
              }else{
                  phi[i * primes[j]] = phi[i] * (primes[j] - 1);
              }
          }
      }
  
      LL res = accumulate(phi + 1, phi + 1 + n, 0ll);
      return res;
  }
  ```

  ***

  

- 欧拉定理
  $$
  a^{\phi(n)} \equiv 1 \pmod n \quad (a和n互质)
  $$
  
- 费马小定理
  $$
  a^{n - 1} \equiv 1 \pmod n \quad (n是质数)
  $$
  

***



### 快速幂

```c++
LL qmi(LL a, LL k, LL p)
{
    int res = 1;
    while(k){
        if(k & 1) res = res * a % p;
        k >>= 1;
        a = a * a % p;
    }
    return res;
}
```



***



### 乘法逆元

$$
\frac {a}{b} \equiv a*b^{-1}\pmod P\\
a \equiv a * b ^ {-1} * b\pmod P\\
b^{-1} b \equiv 1 \pmod P\\
若P为质数，b与P互质，使用费马小定理\\
b^{\phi(p)} \equiv 1 \pmod P\\
b^{P-1}\equiv1 \pmod P\\
b*b^{P-2}\equiv1 \pmod P\\
b^{-1}=b^{P-2}\\
$$

一般P为质数直接用费用小定理即可， 其余情况要用扩展欧几里得

***



### 扩展欧几里得

裴蜀定理
$$
ax + by = gcd（a，b）
$$

```c++
//求系数x, y
int exgcd(int a, int b, int &x, int &y)
{
    if(!b){
        x = 1, y = 0;
        return a;
    } 
    
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    
    return d;
}
```



***



### 中国剩余定理

$$
\left\{\begin{array}\\x \equiv a_1 \pmod {m_1} \\  x \equiv a_2 \pmod {m_2} \\.\\.\\.  \\  x \equiv a_k \pmod {m_k}\end{array}\right.\\
m_1,m_2...m_k两两互质\\
令\quad M=m_1*m_2*...*m_k,  \quad M_i = \frac {M}{m_i}\\
则 \quad x = \sum_{i=1}^k {a_i \cdot M_i \cdot M_i^{-1}}
$$

```c++
/*
AcWing 204. 表达整数的奇怪方式
给定 2n 个整数 a1,a2,…,an 和 m1,m2,…,mn，求一个最小的非负整数 x，满足 ∀i∈[1,n],x≡mi(mod ai)。
数据范围
1≤ai≤2^31−1,
0≤mi<ai
1≤n≤25
PS:这题不能直接用中国剩余定理没有互质条件，数学好难
*/
#include <bits/stdc++.h>

using namespace std;
using LL = long long;

const int N = 30;

LL exgcd(LL a, LL b, LL &x, LL &y){
    if(b == 0){
        x = 1, y = 0;
        return a;
    }

    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

LL inline mod(LL a, LL b)
{
    return ((a % b) + b) % b;
}

int main()
{
    int n;
    cin >> n;
    
    LL a1, m1;
    scanf("%lld%lld", &a1, &m1);
    bool has_answer = true;

    for(int i = 1; i < n; i ++){
        LL a2, m2;
        scanf("%lld%lld", &a2, &m2);
        LL k1, k2;
        LL d = exgcd(a1, a2, k1, k2);
        
        if((m2 - m1) % d){
            has_answer = false;
            break;
        }
        k1 = mod(k1 * (m2 - m1) / d, abs(a2 / d));
        m1 = k1 * a1 + m1;
        a1 = abs(a1 / d * a2);
    }
    
    if(has_answer) printf("%lld\n",m1);
    else puts("-1");
    return 0;
}
```

***



### 高斯消元

```c++
//n个方程组有n个未知数
#include <bits/stdc++.h>

using namespace std;

const int N = 110;
·const double eps = 1e-6;
double a[N][N];
int n;

int gauss()
{
    int r, c;
    for(r = 0, c = 0; c < n; c ++){
        int t = r;
        for(int i = r + 1; i < n; i ++)//找绝对值最大的行
            if(fabs(a[i][c]) > fabs(a[t][c]))
                t = i;
                
        if(fabs(a[t][c]) < eps) continue;
        
        for(int i = 0; i < n + 1; i ++) swap(a[t][i], a[r][i]);//将绝对值最大的行换到最顶端
        for(int i = n; i >= c; i --) a[r][i] /= a[r][c];//将当前行的首位变成1
        for(int i = r + 1; i < n; i ++)//用当前行将下面所有的列消成0
            if(fabs(a[i][c]) > eps)
                for(int j = n; j >= c; j --){
                    a[i][j] -= a[r][j] * a[i][c]; 
                }
        
        r ++;
    }
    
    //0无解， 1唯一解， 2无穷多组解
    for(int i = n - 1; i >= 0; i --)
        for(int j = i + 1; j < n; j ++)
            a[i][n] -= a[i][j] * a[j][n];
    
    if(r < n){
        for(int i = r; i < n; i ++){
            if(fabs(a[i][n]) > eps) return 0;;
        }    
        return 2;
    }
    
    return 1;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setiosflags(ios::fixed) << setprecision(2);
    
    cin >> n;
    for(int i = 0; i < n; i ++)
        for(int j = 0; j < n + 1; j ++)
            cin >> a[i][j];
            
    int t = gauss();
    
    if(t == 0) cout << "No solution" << "\n";
    else if(t == 1){
        for(int i = 0; i < n; i ++){
            if (fabs(a[i][n]) < eps) a[i][n] = 0;  //去掉输出 -0.00 的情况
            cout << a[i][n] << "\n";
        }
    }else{
        cout << "Infinite group solutions" << "\n";
    }
    
    return 0;
}
```

***



### 组合数

- 递推求组合数

  ```c++
  for(int i = 0; i < N; i ++)
          for(int j = 0; j <= i; j ++)
              if(!j) c[i][j] = 1;
              else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
  ```

  

- 预处理求组合数

  ```c++
  //首先预处理出所有阶乘取模的余数fact[N]，以及所有阶乘取模的逆元infact[N]
  //如果取模的数是质数，可以用费马小定理求逆元
  #include <bits/stdc++.h>
  
  using namespace std;
  using LL = long long;
  
  const int N = 100010, mod = 1e9 + 7;
  LL fact[N], infact[N];
  
  int qmi(int a, int k, int p)
  {
      int res = 1;
      while(k){
          if(k & 1) res = (LL)res * a % p;
          k >>= 1;
          a = (LL)a * a % p;
      }
      return res;
  }
  
  int main()
  {
      ios::sync_with_stdio(false);
      cin.tie(nullptr);
      
      int n;
      cin >> n;
      
      fact[0] = infact[0] = 1;
      for(int i = 1; i < N; i ++){
          fact[i] = (LL)fact[i - 1] * i % mod;
          infact[i] = (LL)infact[i - 1] * qmi(i, mod - 2, mod) % mod; 
      }
      
      while(n --){
          int a, b;
          cin >> a >> b;
          
          cout << (LL)fact[a] * infact[b] % mod * infact[a - b] % mod << endl;
      }
      
      return 0;
  }
  ```
  
  

- Lucas定理求组合数
  $$
  C_a^b \equiv C_{a \bmod P} ^ {b \bmod P} \cdot C_{\lfloor \frac {a}{P} \rfloor}^{\lfloor \frac {b}{P} \rfloor} \pmod P
  $$

  ```c++
  #include <bits/stdc++.h>
  
  using namespace std;
  using LL = long long;
  
  int p;
  
  int qmi(int a, int k)
  {
      int res = 1;
      while(k){
          if(k & 1) res = (LL)res * a % p;
          k >>= 1;
          a = (LL)a * a % p;
      }
      return res;
  }
  int C(int a, int b)
  {
      int res = 1;
      for(int i = 1, j = a; i <= b; i ++, j --){
          res = (LL)res * j % p;
          res = (LL)res * qmi(i, p - 2) % p; 
      }
      return res;
  }
  
  int lucas(LL a, LL b)
  {
      if(a < p && b < p) return C(a, b);
      return (LL)C(a % p, b % p) * lucas(a / p, b / p) % p;
  }
  
  int main()
  {
      int n;
      cin >> n;
      
      while(n --){
          LL a, b;
          cin >> a >> b >> p;
          cout << lucas(a, b) << endl;;
      }
      
      return 0;
  }
  ```

  

- 分解质因数加高精度求组合数

  ```c++
  /*
  当我们需要求出组合数的真实值，而非对某个数的余数时，分解质因数的方式比较好用：
      1. 筛法求出范围内的所有质数
      2. 通过 C(a, b) = a! / b! / (a - b)! 这个公式求出每个质因子的次数。 n! 中p的次数是 n / p + n / p^2 + n /
  p^3 + .
      3. 用高精度乘法将所有质因子相乘
  */
  
  #include <bits/stdc++.h>
  
  using namespace std;
  
  const int N = 5010;
  int primes[N], cnt;
  int sum[N];
  bool st[N];
  
  void get_primes(int n)
  {
      for(int i = 2; i <= n; i ++){
          if(!st[i]) primes[cnt ++] = i;
          for(int j = 0; primes[j] <= n / i; j ++){
              st[i * primes[j]] = true;
              if(i % primes[j] == 0) break;
          }
      }
  }
  
  int get(int n, int p)
  {
      int res = 0;
      while(n){
          res += n / p;
          n /= p;
      }
      return res;
  }
  
  vector<int> mul(vector<int> res, int b)
  {
      vector<int> C;
      int t = 0;
      for(int i = 0; i < res.size() || t; i ++){
          if(i < res.size()) t += res[i] * b;
          C.push_back(t % 10);
          t /= 10;
      }
      while(C.size() > 1 && C.back() == 0) C.pop_back();
      return C;
  }
  int main()
  {
      int a, b;
      cin >> a >> b;
      
      get_primes(a);
      
      for(int i = 0; i < cnt; i ++){
          int p = primes[i];
          sum[i] = get(a, p) - get(b, p) - get(a - b, p);
      }
      
      vector<int> res;
      res.push_back(1);
      
      for(int i = 0; i < cnt; i ++)
          for(int j = 0; j < sum[i]; j ++)
              res = mul(res, primes[i]);
      
      for(int i = res.size() - 1; i >= 0; i --) cout << res[i];
      
      return 0;
  }
  ```


***



### 卡特兰数

$$
给定n个0和n个1，它们按照某种顺序排成长度为2n的序列，满足任意前缀中0的个数都不少于1的个数的序列的数量为：\\
C_{2\cdot n}^{n} - C_{2\cdot n}^{n - 1} = \frac{C^{n}_{2 \cdot n}}{n + 1}
$$

***



### 容斥原理

$$
|S_1 \cup S_2 \cup \ldots \cup S_n| = \sum_{1 \leq i \leq n} |S_i| - \sum_{1 \leq j < i \leq n}|S_i \cap S_j| + \ldots +{(-1)}^{m-1}|S_1 \cap S_2 \cap \ldots \cap S_m|
$$

```c++
/*
给定一个整数 n 和 m 个不同的质数 p1,p2,…,pm。
请你求出 1∼n 中能被 p1,p2,…,pm 中的至少一个数整除的整数有多少个。
1≤m≤16,
1≤n,pi≤10^9
*/
#include <bits/stdc++.h>

using namespace std;
using LL = long long;

const int N = 18;
int p[N];

int main()
{
    int n, m;
    cin >> n >> m;
    for(int i = 0; i < m; i ++) cin >> p[i];
    
    int res = 0;
    for(int i = 1; i < 1 << m; i ++){
        int t = 1, cnt = 0;
        for(int j = 0; j < m; j ++)
            if(i >> j & 1){
                cnt ++;
                if((LL)t * p[j] > n){
                    t = -1;
                    break;
                }
                t *= p[j];
            }
        
        if(t != -1){
            if(cnt & 1) res += n / t;
            else res -= n / t;
        }
    }
    
    cout << res;
    
    return 0;
}
```

***



### 整除分块

**求解**
$$
\sum_{i = 1}^{n} \lfloor{\frac{n}{i}} \rfloor
$$
**其值只有$O(\sqrt{n})$种可能，且值是一段一段分布的，对于$i(1 \leq i \leq n), 其所在段的最后一个数字为:$**
$$
\lfloor{\frac{n}{\lfloor {\frac{n}{i}} \rfloor}}\rfloor
$$

***



### 莫比乌斯反演

 **莫比乌斯函数**
$$
\mu(n) = 
\begin{cases}
1 \quad n = 1\\
0 \quad n含有平方因子\\
{(-1)}^k \quad k为n的不同质因子的个数\\
\end{cases}
$$
**莫比乌斯函数的重要性质**
$$
S(n) = 
\sum_{d|n}\mu(n) =
\begin{cases}
1 \quad n = 1\\
0 \quad n \neq 1
\end{cases}
$$
**这一性质的证明**
$$
只需考虑质因子的次数为0或1情况即可\\
\sum_{d|n}\mu(d) = C_{k}^{0}\cdot{(-1)}^{0} +  C_{k}^{1}\cdot{(-1)}^{1} + \ldots + C_{k}^{k}\cdot{(-1)}^{k} = {(-1 + 1)} ^ k = 0
(二项式定理)
$$
**莫比乌斯反演**
$$
若F(n) = \sum_{d|n} f(d), 则f(n) = \sum_{d|n} \mu(n) \cdot F(\frac{n}{d}) \quad 其中d为n的约数
$$
**莫反的证明**
$$
\begin{aligned}
f(n)& = \sum_{d|n} \mu(d) \cdot F(\frac{n}{d}) \\
&= \sum_{d|n} \mu(d) \sum_{i|{\frac{n}{d}}}f(i)\\
&= \sum_{i|n}f(i) \sum_{d|{\frac{n}{i}}}\mu(d) \\
&=  \sum_{i|n}f(i) \cdot S(\frac{n}{i})\\
&= f(n)
\end{aligned}
$$
**通常应用的是另一种形式的莫反**
$$
若F(n) = \sum_{n|d} f(d), 则f(n) = \sum_{n|d} \mu(\frac{d}{n}) \cdot F(d) \quad 其中d为n的倍数
$$
**证明**
$$
\begin{aligned}
f(n)& = \sum_{n|d} \mu(\frac{d}{n}) \cdot F(d) \\
&= \sum_{n|d} \mu(\frac{d}{n}) \sum_{d|i}f(i)\\
&= \sum_{n|i}f(i) \sum_{\frac{d}{n}| \frac {i}{n}}\mu(\frac{d}{n}) \\
&=  \sum_{i|n}f(i) \cdot S(\frac{i}{n})\\
&= f(n)
\end{aligned}
$$




```c++
/*
对于给出的 n 个询问，每次求有多少个数对 (x,y)，满足 a≤x≤b，c≤y≤d，且 gcd(x,y)=k，gcd(x,y) 函数为 x 和 y 的最大公约数。
数据范围均为50000
*/

#include <bits/stdc++.h>

using namespace std;
using LL = long long;

const int N = 50010;
int primes[N], mu[N], cnt, sum[N];
bool st[N];

//线性筛求莫比乌斯函数
void init()
{
    mu[1] = 1;
    for(int i = 2; i < N; i ++){
        if(!st[i]) primes[cnt ++] = i, mu[i] = -1;
        for(int j = 0; primes[j] * i < N; j ++){//这里不能写成 primes[j] < N / i;
            st[i * primes[j]] = true;
            if(i % primes[j] == 0) break;
            mu[i * primes[j]] = - mu[i];
        }
    }

    for(int i = 1; i < N; i ++)
        sum[i] = sum[i - 1] + mu[i];
}

//数论分块
int g(int n, int i)
{
    return n / (n / i);
}

//求莫比乌斯反演中的f(k)
LL f(int a, int b, int k)
{
    a /= k, b /= k;
    LL res = 0;
    int n = min(a, b);
    for(int l = 1, r; l <= n; l = r + 1){
        r = min(g(a, l), g(b, l));
        res += 1ll * (sum[r] - sum[l - 1]) * (a / l) * (b / l);
    }

    return res;
}

void solve()
{   
    int a, b, c, d, k;
    cin >> a >> b >> c >> d >> k;

    LL res = f(b, d, k) - f(a - 1, d, k) - f(b, c - 1, k) + f(a - 1, c - 1, k);
    cout << res << "\n";
} 

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setiosflags(ios::fixed) << setprecision(12);

    init();
    int T = 1;
    cin >> T;
    while(T --) solve();

    return 0;
}
```

***



### 积性函数

**定义**
$$
若一个函数f(x)满足：\forall gcd(a,b) = 1, \quad f(a \cdot b) = f(a) \cdot f(b),则函数f(x)称为积性函数
$$
**常见的积性函数**

- 欧拉函数
- 莫比乌斯函数



***





### BSGS

**求解**
$$
a^x \equiv b \pmod p
$$
**$a$与$p$互质 $O(\sqrt {n})$**
$$
由欧拉定理\\
a^{\phi(n)} \equiv 1 \pmod p \\
\Rightarrow a^{x} \equiv a^{x \bmod {\phi (p)}} \pmod p\\
因此只需要计算a^0, a^1 \ldots a^{\phi{(p)} - 1}即可\\
为了方便计算，计算a^0, a^1 \ldots a^{p}, 其中a^0特判\\
则把1 \ldots p分成k段，k = \sqrt{p} + 1\\
对于每一个值t, t = k \cdot x - y \quad x \in [1, k], y \in [0, k - 1]\\
t的取值范围为[1, k ^ 2], k^2 > p > \phi (p) - 1, \quad 因此1 \ldots \phi (p) - 1的所有值被计算\\
a^t \equiv b \pmod p \\
\Rightarrow a^{k\cdot x - y} \equiv b \pmod p\\
\Rightarrow a^{k \cdot x} \equiv b \cdot a^{y} \pmod p\\
对于b \cdot a ^{y} \pmod p, y\in [0, k -1] 预处理\\
a^{k \cdot x} 查询有无合法的 b \cdot a ^{y} \pmod p即可
$$


```c++
#include <bits/stdc++.h>

using namespace std;
using LL = long long;

int bsgs(int a, int b, int p)
{   
    if(p == 1) return 0;
    if(1 == b % p) return 0;

    unordered_map<int, int> mp; 
    int k = sqrt(p) + 1;

    //预处理
    for(LL i = 0, j = b % p; i <= k - 1; i ++, j = j = j * a % p){
        mp[j] = i;
    }

    LL ak = 1;
    for(int j = 1; j <= k; j ++) ak = ak * a % p;

    for(int i = ak, j = 1; j <= k; j ++, i = i * ak % p){
        if(mp.count(i)) return k * j - mp[i];
    }

    return -1;
}
 
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setiosflags(ios::fixed) << setprecision(12);

    int a, b, p;
    while(cin >> a >> p >> b, a | b | p){
        int res = bsgs(a, b, p);
        if(res == -1) cout << "No Solution" << "\n";
        else cout << res << "\n";
    }

    return 0;
}
```



**$a$与$p$不一定互质**
$$
令d = \gcd(a, p)\\
1、d = 1, 直接bsgs求解即可\\
2、d > 1, 首先特判0，成功返回0即可\\
不成功时：(x > 0) \\
a^x \equiv b \pmod p\\
\Rightarrow a ^ x + k \cdot p = b \quad(若b不能整除d, 无解)\\
\Rightarrow a \cdot a ^ {x - 1} + k \cdot p = b\\
\frac {a}{d} \cdot a^{x - 1} + k \cdot \frac {p}{d} = \frac{b}{d}\\
等价于求解 ： \frac {a}{d} \cdot a ^ {x - 1} \equiv \frac {b}{d} \pmod {\frac {p}{d}}\\
\Rightarrow a ^ {x - 1} \equiv \frac {b}{d} \cdot (\frac {a}{d})^{-1} \pmod {\frac {p}{d}}\\
求解逆元用扩展欧几里得求\\
若a 与 {\frac {p}{d}}不互质， 重复上述过程，互质用bsgs求解
$$


```c++
#include <bits/stdc++.h>

using namespace std;
using LL = long long;

const int INF = 0x3f3f3f3f;

int exgcd(int a, int b, int &x, int &y)
{
    if(!b){
        x = 1, y = 0;
        return a;
    }

    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;

    return d;
}

int bsgs(int a, int b, int p)
{   
    if(p == 1) return 0;
    if(1 == b % p) return 0;

    unordered_map<int, int> mp; 
    int k = sqrt(p) + 1;

    //预处理
    for(LL i = 0, j = b % p; i <= k - 1; i ++, j = j = j * a % p){
        mp[j] = i;
    }

    LL ak = 1;
    for(int j = 1; j <= k; j ++) ak = ak * a % p;

    for(int i = ak, j = 1; j <= k; j ++, i = i * ak % p){
        if(mp.count(i)) return k * j - mp[i];
    }

    return -INF;
}

int exbsgs(int a, int b, int p)
{   
    b = (b % p + p) % p;

    int d = __gcd(a, p);
    if(d > 1){
        if(1 == b % p) return 0;
        if(b % d) return -INF;
        int x, y;
        exgcd(a / d, p / d, x, y);
        return exbsgs(a, 1ll * b / d * x % (p / d), p / d) + 1;

    }else{
        return bsgs(a, b, p);
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setiosflags(ios::fixed) << setprecision(12);

    
    int a, b, p;
    while(cin >> a >> p >> b, a | p | b){
        int res = exbsgs(a, b, p);
        if(res < 0) cout << "No Solution" << "\n";
        else cout << res << "\n";
    }

    return 0;
}
```





****

# 杂项

## 矩阵快速幂

```c++
/*
求斐波那契数列前n项和
*/

#include <bits/stdc++.h>

using namespace std;
using LL = long long;

const int N = 3;
int F[N][N] = {
    {1, 1, 1},
    {0, 0, 0},
    {0, 0, 0},
};
int A[N][N] = {
    {0, 1, 0},
    {1, 1, 1},
    {0, 0, 1},
};
int n, mod;

void mul(int ans[][N], int a[][N], int b[][N])
{
    int temp[N][N] = {0};
    for(int i = 0; i < N; i ++)
        for(int j = 0; j < N; j ++)
            for(int k = 0; k < N; k ++)
                temp[i][j] = (temp[i][j] + 1ll * a[i][k] * b[k][j]) % mod;
                
    memcpy(ans, temp, sizeof(temp));
}

void qmi(int k)
{
    while(k){
        if(k & 1) mul(F, F, A);
        mul(A, A, A);
        k >>= 1;
    }    
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> n >> mod;
    
    qmi(n - 1);
    
    cout << F[0][2] << "\n";
    
    return 0;
}
```

