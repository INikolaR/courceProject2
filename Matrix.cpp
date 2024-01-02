#include "Matrix.h"

Matrix::Matrix(std::vector<std::vector<double>> v) : data_(v) {}

Matrix::Matrix(std::size_t n, std::size_t m) {
    data_ = std::vector<std::vector<double>> (n, std::vector<double> (m, 0.0));
}

Matrix::Matrix(const Matrix & other) : data_(other.data_) {}

Matrix::Matrix(Matrix && other) noexcept : data_(std::forward<std::vector<std::vector<double> > >(other.data_)){}

Matrix &Matrix::operator=(const Matrix & other) {
    return *this = Matrix(other);
}

Matrix &Matrix::operator=(Matrix && other) noexcept {
    Matrix tmp = std::move(other);
    std::swap(data_, tmp.data_);
    return *this;
}

std::vector<std::vector<double> > Matrix::toVec() {
    return data_;
}

std::pair<std::size_t, std::size_t> Matrix::size() const {
    if (data_.empty()) {
        return {0, 0};
    }
    return std::make_pair(data_.size(), data_[0].size());
}

std::vector<double> &Matrix::operator[](int i) {
    return data_[i];
}

const std::vector<double> &Matrix::operator[](int i) const {
    return data_[i];
}

Matrix::~Matrix() {

}

Matrix operator+(const Matrix & m1, const Matrix & m2) {
    std::pair<std::size_t, std::size_t> sz1 = m1.size();
    if (sz1 != m2.size()) {
        throw new std::exception();
    }
    Matrix sum = Matrix(sz1.first, sz1.second);
    for (int i = 0; i < sz1.first; ++i) {
        for (int j = 0; j < sz1.second; ++i) {
            sum[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return sum;
}

Matrix operator-(const Matrix &m1, const Matrix &m2) {
    std::pair<std::size_t, std::size_t> sz1 = m1.size();
    if (sz1 != m2.size()) {
        throw new std::exception();
    }
    Matrix sub = Matrix(sz1.first, sz1.second);
    for (int i = 0; i < sz1.first; ++i) {
        for (int j = 0; j < sz1.second; ++i) {
            sub[i][j] = m1[i][j] - m2[i][j];
        }
    }
    return sub;
}

Matrix operator-(const Matrix &m) {
    return Matrix(m.size().first, m.size().second) - m;
}

Matrix operator*(const Matrix &m1, const Matrix &m2) {
    std::pair<std::size_t, std::size_t> sz1 = m1.size();
    std::pair<std::size_t, std::size_t> sz2 = m2.size();
    if (sz1.second != sz2.first) {
        throw new std::exception();
    }
    Matrix mul = Matrix(sz1.first, sz2.second);
    for (int i = 0; i < sz1.first; ++i) {
        for (int j = 0; j < sz2.second; ++j) {
            double sum = 0;
            for (int k = 0; k < sz1.second; ++k) {
                sum += m1[i][k] * m2[k][j];
            }
            mul[i][j] = sum;
        }
    }
    return mul;
}
