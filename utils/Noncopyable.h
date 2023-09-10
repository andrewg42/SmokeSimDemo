#pragma once

struct Noncopyable {
    Noncopyable() = default;
    virtual ~Noncopyable() = default;
    Noncopyable(Noncopyable const &) = delete;
    Noncopyable operator=(Noncopyable const &) = delete;
};
