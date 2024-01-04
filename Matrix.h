#pragma once

#include <array>

template <std::size_t N, std::size_t M>
class Matrix {
public:
    explicit Matrix(std::array<std::array<double, M>, N> v): data_(v) {}
    Matrix<N, M>() = default;
    Matrix<N, M>(const Matrix<N, M>& other) : data_(other.data_) {}
    Matrix<N, M>(Matrix<N, M>&& other) noexcept : data_(std::forward<std::array<std::array<double, M>, N>>(other.data_)){}
    Matrix<N, M>& operator=(const Matrix<N, M>& other) {
        return *this = Matrix(other);
    }
    Matrix<N, M>& operator=(Matrix<N, M>&& other) noexcept {
        Matrix tmp = std::move(other);
        std::swap(data_, tmp.data_);
        return *this;
    }
    std::array<double, M>& operator[](int i) {
        return data_[i];
    }
    const std::array<double, M>& operator[](int i) const {
        return data_[i];
    }
    ~Matrix<N, M>() {};

    std::array<std::array<double, M>, N> toArray() {
        return data_;
    }
    std::pair<std::size_t, std::size_t> size() const {
        return std::make_pair(N, M);
    }

    Matrix<M, N> transposed() {
        Matrix<M, N> transposed = Matrix<M, N>();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                transposed[j][i] = *this[i][j];
            }
        }
        return transposed;
    }

    Matrix<N, M> forAll(double (*func)(double)) {
        Matrix<N, M> cp = *this;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                cp[i][j] = func(cp[i][j]);
            }
        }
        return cp;
    }
private:
    std::array<std::array<double, M>, N> data_ = std::array<std::array<double, M>, N> ();
};

template <std::size_t N, std::size_t M>
Matrix<N, M> operator+(const Matrix<N, M> & m1, const Matrix<N, M> & m2) {
    Matrix sum = Matrix<N, M>();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            sum[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return sum;
}

template <std::size_t N, std::size_t M>
Matrix<N, M> operator-(const Matrix<N, M> &m1, const Matrix<N, M> &m2) {
    Matrix sub = Matrix<N, M>();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            sub[i][j] = m1[i][j] - m2[i][j];
        }
    }
    return sub;
}

template <std::size_t N, std::size_t M>
Matrix<N, M> operator-(const Matrix<N, M> &m) {
    return Matrix(m.size().first, m.size().second) - m;
}

template <std::size_t N, std::size_t M, std::size_t K>
Matrix<N, K> operator*(const Matrix<N, M> &m1, const Matrix<M, K> &m2) {
    Matrix mul = Matrix<N, K>();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            double sum = 0;
            for (std::size_t j = 0; j < M; ++j) {
                sum += m1[i][j] * m2[j][k];
            }
            mul[i][k] = sum;
        }
    }
    return mul;
}

template <std::size_t N, std::size_t M>
Matrix<N, M> operator^(const Matrix<N, M> &m1, const Matrix<N, M>& m2) {
    Matrix<N, M> mul = Matrix<N, M>();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            mul = m1[i][j] * m2[i][j];
        }
    }
    return mul;
}

template <std::size_t N, std::size_t M>
Matrix<N, M> operator*(double c, const Matrix<N, M>& m) {
    Matrix<N, M> mul = Matrix<N, M>();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            mul = c * m[i][j];
        }
    }
    return mul;
}
