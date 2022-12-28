#pragma once 

#include <memory>
#include <vector>

template <typename T>
class RingBuffer
{
public:
    RingBuffer(int size) : vec(size), readIt(vec.begin()), writeIt(vec.begin()) {}

    template<typename... Ts>
    void insert(Ts... args)
    {
        *writeIt = std::make_unique<T>(args...);
        writeIt++;
        readIt = writeIt - 1;

        if (writeIt == vec.end())
            writeIt = vec.begin();
    }

    T& current()
    {
        return *readIt->get();
    }

    T& prev()
    {
        if (readIt == vec.begin())
            return *(vec.end() - 1)->get();

        return *(readIt - 1)->get();
    }

    std::size_t size()
    {
        int size = 0;
        for (std::unique_ptr<T>& element : vec)
        {
            if (element)
                size++;
        }

        return size;
    }

private:
    std::vector<std::unique_ptr<T>> vec;
    typename std::vector<std::unique_ptr<T>>::iterator readIt, writeIt;
};