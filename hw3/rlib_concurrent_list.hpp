#ifndef RLIB_CONCURR_LIST_HPP_
#define RLIB_CONCURR_LIST_HPP_

#include <atomic>
#include <iterator>

namespace rlib {
    namespace concurrency {
        template <typename T>
        class single_list {
            struct node {
                T data;
                std::atomic<node *> next;
                node(T &&data, node *next = nullptr) : data(data), next(next) {}
            };
        public:
            using element_type = T;
            using this_type = single_list<T>;

            single_list() : possible_tail(nullptr), root(nullptr) {}

            void push_back(T &&ele) {
                node *new_node = new node(std::forward<T>(ele));
                node *null_ = nullptr;
                // DO NOT allow pop_back.
                if(possible_tail == nullptr) {
                    if(std::atomic_compare_exchange_strong(&possible_tail, &null_, new_node)) {
                        root = new_node;
                        return;
                    }
                }
                while(!std::atomic_compare_exchange_weak(&(possible_tail.load()->next), &null_, new_node));
                possible_tail.store(new_node);
            }

        public:
            class iterator : std::forward_iterator_tag {
                friend class single_list<T>;
            public:
                using pointer = T *;
                using reference = T &;
                using this_type = iterator;
                explicit iterator(node * ptr) : ptr(ptr) {}

                reference operator*() const {
                    // If this is an iterator to empty_list.begin(), then nullptr->data throws.
                    return ptr->data;
                }
                pointer operator->() const {
                    // If this is an iterator to empty_list.begin(), then nullptr->data throws.
                    return &ptr->data;
                }
                this_type &operator++() {
                    ptr = ptr->next;
                    return *this;
                }
                const this_type operator++(int) {
                    iterator backup(ptr);
                    operator++();
                    return std::move(backup);
                }
                bool operator==(const this_type &another) const {
                    return ptr == another.ptr;
                }

                bool operator!=(const this_type &another) const {
                    return !operator==(another);
                }
            private:
                node * ptr;
            };

            class const_iterator : std::forward_iterator_tag {
                friend class single_list<T>;
            public:
                using pointer = const T *;
                using reference = const T &;
                using this_type = const_iterator;
                explicit const_iterator(const node * ptr) : ptr(ptr) {}

                reference operator*() const {
                    // If this is an iterator to empty_list.begin(), then nullptr->data throws.
                    return ptr->data;
                }
                pointer operator->() const {
                    // If this is an iterator to empty_list.begin(), then nullptr->data throws.
                    return &ptr->data;
                }
                this_type &operator++() {
                    ptr = ptr->next;
                    return *this;
                }
                const this_type operator++(int) {
                    iterator backup(ptr);
                    operator++();
                    return std::move(backup);
                }
                bool operator==(const this_type &another) const {
                    return ptr == another.ptr;
                }

                bool operator!=(const this_type &another) const {
                    return !operator==(another);
                }
            private:
                const node * ptr;
            };

            iterator begin() {
                return iterator(root);
            }
            iterator end() {
                return iterator(nullptr);
            }
            const_iterator begin() const {
                return const_iterator(root);
            }
            const_iterator end() const {
                return const_iterator(nullptr);
            }
            const_iterator cbegin() const {
                return const_iterator(root);
            }
            const_iterator cend() const {
                return const_iterator(nullptr);
            }


        private:
            node * root;
            volatile std::atomic<node *> possible_tail;
        };
    }
}

#endif