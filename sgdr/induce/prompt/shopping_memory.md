## Task: I have a lot of Nintendo Switch game cards now, help me find the best storage option to fit all {{num}} cards

### Example 1: I have a lot of Nintendo Switch game cards now, help me find the best storage option to fit all 11 cards
To find the best storage option for 11 Nintendo Switch game cards, I need to search for storage cases specifically designed for Nintendo Switch game cards. I will use the search bar to look for "Nintendo Switch game card storage case".
```
click('567')
fill('567', 'Nintendo Switch game card storage case')
press('567', 'Enter')
```
I can see several storage options for Nintendo Switch game cards. I will look for the best option that can fit all 11 cards. The "HEIYING Game Card Case" has 24 game card slots and seems to be a good fit. I will select this option
```click('1768')```

### Example 2: I have a lot of Nintendo Switch game cards now, help me find the best storage option to fit all 31 cards
To find the best storage option for 31 Nintendo Switch game cards, I need to search for storage solutions specifically designed for Nintendo Switch game cards. I will start by using the search bar to look for "Nintendo Switch game card storage".
```
click('179')
fill('179', 'Nintendo Switch game card storage')
press('179', 'Enter')
```

To find the best storage option for 31 Nintendo Switch game cards, I need to review the available products on the search results page. I will start by examining the details of the first few products listed to see their storage capacities.
```scroll(0, 300)```

To find the best storage option for 31 Nintendo Switch game cards, I need to look at the details of the storage cases listed on the search results page. I will start by examining the first few options to see their capacity.
```click('1308')```

The current page shows a product that can store up to 40 Nintendo Switch game cards, which is sufficient for the user's requirement of storing 31 cards. I will inform the user about this product.
```send_msg_to_user('The "Game Card Holder Storage Case for Nintendo Switch Games or PS Vita Game Case or SD Memory Cards, Black" can store up to 40 Nintendo Switch game cards, which will fit all 31 of your cards. The price is $11.99.')```

## Reusable Workflows
"""
Task: Search for a product in the search bar.
Action Trajectory:
```click('567')```  # click on the search bar
```fill('567', {product_name})```  # fill the search bar with the product name
```keyboard_press("Enter")```  # press Enter to search for the product
"""


## Show me the {{info}} for order number {{order_number}}.
### Example 1: Show me the billing address for order number 00178.
Click on the "My Account" link to access the account section where I can look for order information.
```click("227")```

Click the "View All" link with bid 1742 next to "Recent Orders". This should take me to a page showing all orders where I can look for order #00178.
```click("1742")```

Click "Page 2" link with bid 1816 to see the next set of orders, where I can continue looking for order #00178.
```click("1816")```

Click "Page 3" link with bid 1816 to see the next set of orders, where I can continue looking for order #00178.
```click("1824")```

I can see order #00178 with its "View Order" link (bid 1701). I need to click this link to view the order details and billing address, but need to format the action correctly. This will take me to the order details page where I can find the billing address information.
```click("1701")```

I need to send this information to the user, using the `send_msg_to_user` action with proper formatting.
```send_msg_to_user("The billing address for order #00178 is: Emma Lopez, 101 S San Mateo Dr, San Mateo, California, 94010, United States")```

## Reusable Workflows
"""
Task: Navigate from homepage to order history page.
Action Trajectory:
```click('227')```  # Click on the "My Account" link to access the account section where I can look for order information.
```click("1742")```  # Click the "View All" link with bid 1742 next to "Recent Orders". This should take me to a page showing all orders where I can look for order #00178.
"""