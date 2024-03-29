#pragma once


namespace fastertransformer {

template <class T>
class AverageMeter {
public:
    AverageMeter() : val_(0), avg_(0), sum_(0), count_(0) {}

    void reset()
    {
        val_ = 0;
        avg_ = 0;
        sum_ = 0;
        count_ = 0;
    }

    void update(const T& val)
    {
        val_ = val;
        sum_ += val;
        count_ += 1;
        avg_ = sum_ / count_;
    }

    T getAvg() const
    {
        return avg_;
    }
          
private:
    T val_;
    T avg_;
    T sum_;
    size_t count_;
};

} // fastertransformer