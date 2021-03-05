/*******************************************************************************
 * This file is part of Mitos.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>

#include "../include/titles.h"

/* Returns the id of the title in the array. Returns -1 if not found */
int findTitle(char **titles, const char *title, int ntitles) {
    int len = strlen(title);
    int found = -1;

    for (int i=0; i<ntitles; i++) {
        if (strlen(titles[i]) != len) continue;
        if (strncmp(titles[i], title, len) == 0) {
            found = i;
        }
    }

    return found;
}
