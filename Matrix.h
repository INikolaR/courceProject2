#include <vector>

class Matrix {
public:
    explicit Matrix(std::vector<std::vector<double> > v);
    Matrix(std::size_t n, std::size_t m);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    std::vector<double>& operator[](int i);
    const std::vector<double>& operator[](int i) const;
    ~Matrix();

    std::vector<std::vector<double> > toVec();
    std::pair<std::size_t, std::size_t> size() const;
private:
    std::vector<std::vector<double> > data_;
};

Matrix operator+(const Matrix& m1, const Matrix& m2);
Matrix operator-(const Matrix& m);
Matrix operator-(const Matrix& m1, const Matrix& m2);
Matrix operator*(const Matrix& m1, const Matrix& m2);