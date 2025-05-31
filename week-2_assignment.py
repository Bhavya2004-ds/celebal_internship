class Node:
    None

class LinkedList:
    def create_list(self):
        self.head = None

    def create_node(self, data):
        node = Node()
        node.data = data
        node.next = None
        return node

    def add_node(self, data):
        if not hasattr(self, 'head'):
            self.create_list()

        new_node = self.create_node(data)
        if self.head == None:
            self.head = new_node
        else:
            temp = self.head
            while temp.next != None:
                temp = temp.next
            temp.next = new_node

    def print_list(self):
        if not hasattr(self, 'head') or self.head == None:
            print("List is empty.")
        else:
            temp = self.head
            while temp != None:
                print(temp.data, end=" -> ")
                temp = temp.next
            print("None")

    def delete_nth_node(self, n):
        try:
            if not hasattr(self, 'head') or self.head == None:
                print("Error: list is empty.")
                return
            if n <= 0:
                print("Error: index must be 1 or more.")
                return
            if n == 1:
                self.head = self.head.next
                return
            temp = self.head
            count = 1
            while temp != None and count < n - 1:
                temp = temp.next
                count += 1
            if temp == None or temp.next == None:
                print("Error: index out of range.")
                return
            temp.next = temp.next.next
        except:
            print("Something went wrong.")


my_list = LinkedList()

my_list.add_node(10)
my_list.add_node(20)
my_list.add_node(30)
my_list.add_node(40)

print("Original List:")
my_list.print_list()

print("\nDeleting 2nd node:")
my_list.delete_nth_node(2)
my_list.print_list()

print("\nDeleting 1st node:")
my_list.delete_nth_node(1)
my_list.print_list()

print("\nTrying to delete from empty list:")
empty = LinkedList()
empty.delete_nth_node(1)

print("\nTrying to delete node at position 10:")
my_list.delete_nth_node(10)
