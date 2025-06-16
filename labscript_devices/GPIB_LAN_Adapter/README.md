#  GPIB ETHERNET ADAPTER implementation

# Stand (16.06.2025)
Currently on hold, but we may revisit it after the implementation of the HP devices with the Kofortrinic adapter. This implementation will provide a more flexible approach for managing GPIB devices.

## TODO
1.  Label the adapter , has fix ip adress now
2.  Write communication methodedes between adapter and slave 
3.  Description PROLOGIX


### BW
- save slave device parameters in the transition methodes

### BT

### LD

### D
- if error wrapper gets siblings , we move them in an util house
- understand why error wrapper interfers with get_adress_gpib
- _recv : check buffer story
- Make Adapter felxible to different kinds of terminations 
        --> Requires communication between Adapter class and slave class in the blacs wroker (see problem 2)

### Slave
- @property for limits

- IDEA: allow user to add to a categorie of commands
    --> Purpose : device commands are organized


## Current possible utilization
TODO

### configuration
TODO


##  Example Script

### In the Python connection table

```python
TODO()
```

### In the python experiment file
* `TODO( )` 

```python
start()
t = 0
    # TODO
stop(t)
```