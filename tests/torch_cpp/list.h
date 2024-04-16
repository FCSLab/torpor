#ifndef _LIST_H_
#define _LIST_H_
#include <unistd.h>

typedef struct list_t {
    void *elements;
    size_t length;
    size_t capacity;
    size_t element_size;
} list;

int list_init(list *l, size_t element_size);
int list_free(list *l);
int list_free_elements(list *l);

/** appends an element to the list.
 * the list is extended if needed.
 * @l the list
 * @new_element the new list element
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_append(list *l, void **new_element);

/** appends an element and copies its content.
 * the new element is copied using element_size as set during initialization.
 * @l the list
 * @new_element element from which to copy
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_append_copy(list *l, void *new_element);

/** return list element at the specified index.
 * @l the list
 * @at the index of the element
 * @element a pointer which should point to the element
 * @return 0 on success; 1 if l is NULL or at is larger then the
 *         list length
 */
int list_at(list *l, size_t at, void **element);

/** returns the position at which the requested element should be.
 * performs no checks - use with caution
 * @l the list
 * @at the index of the element
 * @return the element address or NULL on error
 */
void *list_get(list *l, size_t at);

/** removes an element from the list.
 * if element is NULL, the element is freed.
 * @l the list
 * @at the index of the element to be removed
 * @element the element that was removed in returned in this pointer.
 * @return 0 on success; 1 if l is NULL or at is larger then the list
 *         length
 */
int list_rm(list *l, size_t at);

/** inserts an element at the given position and copies its content.
 * the new element is copied using element_size as set during initialization.
 * Elements with indexes greater than @at are moved by one index.
 * @l the list
 * @at the position of the new element
 * @new_element element from which to copy
 * @return 0 on success; 1 if l is NULL or allocation failed
 */
int list_insert(list *l, size_t at, void *new_element);

#define INITIAL_CAPACITY 4

int list_init(list *l, size_t element_size)
{
    if (l == NULL) {
        return 1;
    }
    if (element_size == 0LL) {
        return 1;
    }
    memset(l, 0, sizeof(list));
    if ((l->elements = malloc(INITIAL_CAPACITY*element_size)) == NULL) {
        return 1;
    }
    l->element_size = element_size;
    l->capacity = INITIAL_CAPACITY;
    l->length = 0LL;

    return 0;
}

int list_free(list *l)
{
    if (l == NULL) {
        return 1;
    }
    free(l->elements);
    l->length = 0;
    l->capacity = 0;
    return 0;
}

int list_free_elements(list *l)
{
    if (l == NULL) {
        return 1;
    }
    for (size_t i=0; i < l->length; ++i) {
        free(*(void**)list_get(l, i));
    }
    return 0;
}

int list_append(list *l, void **new_element)
{
    int ret = 0;
    if (l == NULL) {
        return 1;
    }
    if (l->capacity == l->length) {
        l->elements = realloc(l->elements, l->capacity*2*l->element_size);
        if (l->elements == NULL) {
            /* the old pointer remains valid */
            return 1;
        }
        l->capacity *= 2;
    }
    if (new_element != NULL) {
        *new_element = list_get(l, l->length++);
    }

    return ret;
}

int list_append_copy(list *l, void *new_element)
{
    int ret = 0;
    void *elem;
    if ( (ret = list_append(l, &elem)) != 0) {
        goto out;
    }
    memcpy(elem, new_element, l->element_size);
 out:
    return ret;
}

int list_at(list *l, size_t at, void **element)
{
    if (l == NULL) {
        return 1;
    }
    if (at >= l->length) {
        return 1;
    }
    if (element != NULL) {
        *element = list_get(l, at);
    }
    return 0;
}

inline void* list_get(list *l, size_t at) {
    return (l->elements+at*l->element_size);
}

int list_insert(list *l, size_t at, void *new_element)
{
    if (l == NULL) {
        return 1;
    }
    if (at > l->length) {
        return 1;
    }
    if (at == l->length) {
        return list_append_copy(l, new_element);
    }

    if (list_append(l, NULL) != 0) {
        return 1;
    }
    memmove(list_get(l, at+1), list_get(l, at), (l->length-at)*l->element_size);

    if (new_element != NULL) {
        memcpy(list_get(l, at), new_element, l->element_size);
    }

    l->length += 1; //appending a NULL element does not increase list length
    return 0;
}

int list_rm(list *l, size_t at)
{
    if (l == NULL) {
        return 1;
    }
    if (at >= l->length) {
        return 1;
    }
    if (at < l->length-1) {
        memmove(list_get(l, at), list_get(l, at+1), (l->length-1-at)*l->element_size);
    }
    l->length -= 1;
    return 0;
}


#endif //_LIST_H_
