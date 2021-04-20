#!/usr/bin/bash

#DA Copyright (C). All services registered

# This bash script removes windows line's ending

# $1 -> file name

sed -i 's/\r/ /g' $1